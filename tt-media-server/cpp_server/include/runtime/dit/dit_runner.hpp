// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runtime/dit/dit_types.hpp"

namespace tt::dit {

/**
 * What the C++ worker drives, in this order:
 *   openDevice()   fabric + mesh
 *   warmup()       build the pipeline, run one short generation to compile
 *   generate()...  validate, run, export
 *   close()
 * Implemented by DiTRunner (one host) and LockstepRunner (rank 0 of several
 * hosts, which wraps a DiTRunner).
 */
class IDiTRunner {
 public:
  virtual ~IDiTRunner() = default;
  virtual void openDevice() = 0;
  virtual void warmup() = 0;
  /** Refuses (RequestError) what the model cannot run, before anything is
   *  sent to other ranks or to the mesh. The default accepts everything. */
  virtual void validate(const DiTRequest& request) { (void)request; }
  /** Throws RequestError for a request the model refuses and ExportError when
   *  the finished output cannot be written. */
  virtual DiTResult generate(const DiTRequest& request) = 0;
  /** validate() + the pipeline call, without exporting: what the other ranks
   *  of a multihost mesh run (only rank 0 keeps the output). */
  virtual void infer(const DiTRequest& request) = 0;
  virtual void close() = 0;
};

}  // namespace tt::dit
