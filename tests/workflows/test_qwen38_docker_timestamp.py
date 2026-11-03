"""Pure stdlib tests runnable by the actual Python3.10 CI host interpreter."""
from datetime import datetime
import unittest
from workflows.qwen38_selected_precision import docker_timestamp_epoch,verify_adapter_marker,selected_policy
class DockerTimestampTests(unittest.TestCase):
 def test_actual_ci_nanoseconds_and_short_fractions(self):
  expected=datetime.fromisoformat('2026-11-03T05:15:13.679142+00:00').timestamp()
  self.assertEqual(docker_timestamp_epoch('2026-11-03T05:15:13.679142093Z'),expected)
  self.assertEqual(docker_timestamp_epoch('2026-11-03T05:15:13.679142093+00:00'),expected)
  self.assertEqual(docker_timestamp_epoch('2026-11-03T06:15:13.679142093+01:00'),expected)
  for fraction in ('1','12','123','1234','12345','123456','1234567','12345678','123456789'):
   expected=datetime.fromisoformat('2026-11-03T05:15:13.'+fraction[:6].ljust(6,'0')+'+00:00').timestamp()
   self.assertEqual(docker_timestamp_epoch('2026-11-03T05:15:13.'+fraction+'Z'),expected)
 def test_timezone_and_bad_forms(self):
  self.assertEqual(docker_timestamp_epoch('2026-11-03T06:15:13+01:00'),docker_timestamp_epoch('2026-11-03T05:15:13Z'))
  for value in ('2026-11-03T05:15:13','2026-11-03T05:15:13.1234567890Z','bad','2026-99-03T05:15:13Z'):
   with self.assertRaises(ValueError):docker_timestamp_epoch(value)
 def test_actual_full_marker_keeps_identity_checks(self):
  marker=f'2026-11-03T05:15:13.679142093Z (EngineCore_DP0 pid=603) AUTOPORT_ADAPTER /actual/models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py:Qwen38ForCausalLM layers={list(range(64))} precision={selected_policy()}'
  proof=verify_adapter_marker(marker);self.assertEqual(proof['engine_pid'],603);self.assertEqual(proof['layer_count'],64)
  for bad in (marker.replace('layers=[0, 1,','layers=[1,'),marker.replace('Qwen38ForCausalLM','Other'),marker.replace("'head': 'bfloat8_b'","'head': 'bfloat16'")):
   with self.assertRaises(RuntimeError):verify_adapter_marker(bad)
if __name__=='__main__':unittest.main()
