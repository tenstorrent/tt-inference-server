# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.


import ttnn
from tt_model_runners.base_device_runner import BaseDeviceRunner


class BaseMetalDeviceRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        # Set when the mesh is carved out of a larger parent mesh; see
        # get_parent_mesh_plan. The parent owns the hardware, so it is the
        # handle close_device has to release.
        self._parent_mesh_device = None

    def get_pipeline_device_params(self):
        return None

    def get_parent_mesh_plan(self):
        """Return (parent_shape, submesh_shape, reshape_to) or None.

        Default None means "open settings.device_mesh_shape directly", which is
        right whenever the requested mesh is the whole system.

        On a torus-wired box (BH Galaxy) a partial mesh cannot bring up fabric:
        the routers on the selected chips try to handshake with physical
        neighbours that are outside the mesh, so no partner kernel answers and
        fabric init dies with "Fabric Router Sync: Timeout ... expected status
        0xa2b2c2d2 (LOCAL_HANDSHAKE_COMPLETE)". Measured on g11blx01 with a
        healthy fabric (8 UP eth links on all 32 chips) under FABRIC_1D:

            (4, 8) OK    (8, 4) OK    (1, 32) OK      <- cover all 32 chips
            (1, 4) FAIL  (1, 8) FAIL  (2, 4)  FAIL    <- partial, RouterSync

        FABRIC_1D and FABRIC_1D_RING fail identically on (1, 4), so this is
        about coverage, not ring vs linear topology.

        Returning a plan opens the full system mesh first (every router finds
        its partner), then slices the shape the model wants out of it. This
        mirrors models/tt_dit/tests/models/sd35/run_sd35_submesh.py in
        tt-metal, whose header records the same finding: "Opening a bare 2x2
        mesh on a Galaxy fails fabric router sync (neighbours outside the mesh
        never come up), so open the full system mesh with fabric and slice a
        2x2 submesh out of it."

        reshape_to is the optional relabel that turns a compact block into the
        row a preset expects -- a (2, 2) submesh reshaped to (1, 4) renumbers
        in ring order (device ids 0, 1, 5, 4), which keeps every hop between
        adjacent tp neighbours physical.
        """
        return None

    def set_device(self):
        if self.ttnn_device is None:
            # for now use all available devices
            self.ttnn_device = self._mesh_device()
        self.max_batch_size = self.settings.max_batch_size
        return self.ttnn_device

    def close_device(self):
        try:
            self.logger.info(f"Device {self.device_id}: Closing mesh device...")
            # Submeshes are views onto their parent, so releasing the parent is
            # what frees the hardware. Closing the submesh first would leave the
            # parent holding the devices.
            if self._parent_mesh_device is not None:
                ttnn.close_mesh_device(self._parent_mesh_device)
                self._parent_mesh_device = None
                self.ttnn_device = None
                self.logger.info(
                    f"Device {self.device_id}: Successfully closed parent mesh device"
                )
            elif self.ttnn_device is not None:
                ttnn.close_mesh_device(self.ttnn_device)
                self.logger.info(
                    f"Device {self.device_id}: Successfully closed mesh device"
                )
            else:
                self.logger.info(
                    f"Device {self.device_id}: Device is None, no need to close"
                )
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Failed to close device: {e}")
            raise RuntimeError(
                f"Device {self.device_id}: Device cleanup failed: {str(e)}"
            ) from e

    def get_updated_device_params(self, device_params):
        if device_params is None:
            device_params = {}

        new_device_params = device_params.copy()

        dispatch_core_axis = new_device_params.pop("dispatch_core_axis", None)
        dispatch_core_type = new_device_params.pop("dispatch_core_type", None)
        fabric_tensix_config = new_device_params.get("fabric_tensix_config", None)

        if ttnn.device.is_blackhole():
            # Only when both fabric_config and fabric_tensix_config are set, we can use ROW dispatch, otherwise force to use COL dispatch
            fabric_config = new_device_params.get("fabric_config", None)
            if not (fabric_config and fabric_tensix_config):
                # When not both are set, force COL dispatch
                if dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
                    self.logger.warning(
                        "ROW dispatch requires both fabric and tensix config, using DispatchCoreAxis.COL instead."
                    )
                    dispatch_core_axis = ttnn.DispatchCoreAxis.COL
            elif fabric_config and fabric_tensix_config:
                self.logger.warning(
                    f"Blackhole with fabric_config and fabric_tensix_config enabled, using fabric_tensix_config={fabric_tensix_config}"
                )

        dispatch_core_config = ttnn.DispatchCoreConfig(
            dispatch_core_type, dispatch_core_axis, fabric_tensix_config
        )
        new_device_params["dispatch_core_config"] = dispatch_core_config

        return new_device_params

    def _mesh_device(self):
        try:
            device_ids = ttnn.get_device_ids()
            if not device_ids:
                raise RuntimeError("No TTNN devices available")
            self.logger.info(
                f"Device {self.device_id}: Found {len(device_ids)} available TTNN devices: {device_ids}"
            )

            mesh_shape = ttnn.MeshShape(self.settings.device_mesh_shape)

            device_params = self.get_pipeline_device_params()
            updated_device_params = self.get_updated_device_params(device_params)
            fabric_config = self._configure_fabric(updated_device_params)
            mesh_device = self._initialize_mesh_device(
                mesh_shape, updated_device_params, fabric_config
            )

            self.logger.info(
                f"Device {self.device_id}: Created mesh device with {mesh_device.get_num_devices()} devices"
            )
            return mesh_device
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Device initialization failed: {e}"
            )
            raise RuntimeError(
                f"Unexpected device initialization error: {str(e)}"
            ) from e

    def _configure_fabric(self, updated_device_params):
        return None

    def _initialize_mesh_device(self, mesh_shape, device_params, fabric_config):
        try:
            plan = self.get_parent_mesh_plan()
            if plan is None:
                mesh_device = ttnn.open_mesh_device(
                    mesh_shape=mesh_shape, **device_params
                )
            else:
                parent_shape, submesh_shape, reshape_to = plan
                self.logger.info(
                    f"Device {self.device_id}: opening parent mesh {tuple(parent_shape)} "
                    f"then slicing {tuple(submesh_shape)}"
                    + (f" reshaped to {tuple(reshape_to)}" if reshape_to else "")
                    + " (partial meshes cannot initialize fabric on a torus-wired box)"
                )
                parent = ttnn.open_mesh_device(
                    mesh_shape=ttnn.MeshShape(*parent_shape), **device_params
                )
                try:
                    mesh_device = parent.create_submeshes(
                        ttnn.MeshShape(*submesh_shape)
                    )[0]
                    if reshape_to is not None:
                        mesh_device.reshape(ttnn.MeshShape(*reshape_to))
                except Exception:
                    ttnn.close_mesh_device(parent)
                    raise
                # Keep the parent alive for the life of the submesh.
                self._parent_mesh_device = parent
                self.logger.info(
                    f"Device {self.device_id}: submesh {mesh_device.shape} "
                    f"of parent {parent.shape}"
                )
        except Exception as e:
            try:
                if fabric_config:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception as reset_error:
                self.logger.warning(
                    f"Device {self.device_id}: Failed to reset fabric after device initialization failure: {reset_error}"
                )
            self.logger.error(
                f"Device {self.device_id}: Mesh device initialization failed: {e}"
            )
            raise RuntimeError(f"Mesh device initialization failed: {str(e)}") from e
        return mesh_device
