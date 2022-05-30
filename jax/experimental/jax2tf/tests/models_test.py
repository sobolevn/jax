# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Tests for converting Flax models using jax2tf."""
from absl.testing import absltest

import functools
import numpy as np
import re
import jax
from jax._src import test_util as jtu

from jax.config import config

config.parse_flags_with_absl()

from jax._src import dtypes
import jax.numpy as jnp

import datetime
import os
from typing import Dict, List, Tuple

from jax.experimental.jax2tf.tests.model_harness import ALL_HARNESS_FNS
from jax.experimental.jax2tf.tests.converters import ALL_CONVERTERS

DEFAULT_RTOL = 1e-05


def _write_markdown(results: Dict[str, List[Tuple[str, str,]]]) -> None:
  """Writes all results to Markdown file."""
  table_lines = []
  converters = ALL_CONVERTERS.keys()

  table_lines.append("| Example | " + " ".join([f"{c} |" for c in converters]))
  table_lines.append("|" + (" --- |" * (len(converters) + 1)))

  error_lines = []

  def _header2anchor(header):
    header = header.lower().replace(" ", "-")
    header = "#" + re.sub(r"[^a-zA-Z0-9_-]+", "", header)
    return header

  for example_name, converter_results in results.items():
    # Make sure the converters are in the right order.
    assert [c[0] for c in converter_results] == converters
    line = f"| {example_name} |"
    error_lines.append(f"## `{example_name}`")
    for converter_name, error_msg in converter_results:
      if not error_msg:
        line += " YES |"
      else:
        error_header = (f"Example: `{example_name}` | "
                        f"Converter: `{converter_name}`")
        error_lines.append("### " + error_header)
        error_lines.append(f"```\n{error_msg}\n```")
        error_lines.append("[Back to top](#summary-table)\n")
        line += f" [NO]({_header2anchor(error_header)}) | "
    table_lines.append(line)

  output_path = os.path.join(
      os.path.dirname(__file__), "../g3doc/converters_results.md")
  with open(output_path + ".template") as f_in, open(output_path, "w") as f_out:
    template = f_in.read()
    template = template.replace("{{generation_date}}",
                                str(datetime.date.today()))
    template = template.replace("{{table}}", "\n".join(table_lines))
    template = template.replace("{{errors}}", "\n".join(error_lines))
    f_out.write(template)

  print("Written converter results to %s", output_path)


def _crop_convert_error(msg: str, max_len: int = 250) -> str:
  msg = msg.replace("\\n", "\n").replace("\\t", "\t")

  tflite_msg = "Some ops are not supported by the native TFLite runtime"
  tfjs_msg = "Some ops in the model are custom ops"

  if tflite_msg in msg:
    # Filters out only the missing ops.
    missing_ops = msg.split("\nDetails:\n")[1].split("\n\n")[0]
    return tflite_msg + "\n" + missing_ops

  elif tfjs_msg in msg:
    return msg[msg.index(tfjs_msg):]

  return msg  #if len(msg) < max_len else msg[:max_len] + "... (CROPPED)"


def _get_random_data(x: jnp.ndarray) -> np.ndarray:
  dtype = dtypes.canonicalize_dtype(x.dtype)
  if np.issubdtype(dtype, np.integer):
    return np.random.randint(0, 100, size=x.shape, dtype=dtype)
  elif np.issubdtype(dtype, np.floating):
    return np.array(np.random.uniform(size=x.shape), dtype=dtype)
  elif dtype == np.bool:
    return np.random.choice(a=[False, True], size=x.shape)
  else:
    raise ValueError(f"Unsupported dtype for numerical comparison: {dtype}")


class JaxModelsTest(jtu.JaxTestCase):

  def test_convert_models(self):
    """Tests all converters and write output."""
    results = {}

    for harness_fn in ALL_HARNESS_FNS:
      harness = harness_fn()  # This will create the variables.
      np_assert_allclose = functools.partial(
          np.testing.assert_allclose, rtol=harness.rtol)
      converter_results = []
      for converter in ALL_CONVERTERS:
        print(f"===== Testing example {harness.name}, "
              f"Converter {converter.name}")
        error_msg = ""
        try:
          apply_tf = converter.convert_fn(harness)
          print("=== Conversion OK!")
          if not converter.compare_numerics:
            print("=== Skipping numerical comparison.")
          else:
            xs = [_get_random_data(x) for x in harness.inputs]
            jax_result = harness.apply_with_vars(*xs)
            # We put this in the same try-except clause as the conversion,
            # because TFLite may occasionally throw a runtime error here.
            tf_result = apply_tf(*xs)
        except Exception as e:  # pylint: disable=broad-except
          error_msg = "Conversion error\n" + _crop_convert_error(repr(e))
          print(f"=== {error_msg}")
        else:
          try:
            jax.tree_map(np_assert_allclose, jax_result, tf_result)
            print("=== Numerical comparison OK!")
          except AssertionError as e:
            error_msg = repr(e).replace("\n\n", "\n")  # Output newlines.
            error_msg = "Numerical comparison error:\n" + error_msg
            print(f"=== {error_msg}")
        converter_results.append((converter.name, error_msg))
      results[harness.name] = converter_results

    if os.environ.get("JAX_OUTPUT_MODELS_TEST_DOC"):
      _write_markdown(results)
    else:
      print("=== NOT writing results to Markdown (env variable is not set).")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
