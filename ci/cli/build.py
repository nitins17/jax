#!/usr/bin/python
# A cli using argparse that accepts the enums defined in matrix.py as options
import argparse
import asyncio
import logging
import os
import platform
import collections
import sys
import subprocess
from helpers import command, tools

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ArtifactBuildSpec = collections.namedtuple(
    "ArtifactBuildSpec",
    ["bazel_build_target", "wheel_binary"],
)

ARTIFACT_BUILD_TARGET_DICT = {
    "jaxlib": ArtifactBuildSpec("//jaxlib/tools:build_wheel", "bazel-bin/jaxlib/tools/build_wheel"),
    "jax-cuda-plugin": ArtifactBuildSpec("//jaxlib/tools:build_gpu_kernels_wheel", "bazel-bin/jaxlib/tools/build_gpu_kernels_wheel"),
    "jax-cuda-pjrt": ArtifactBuildSpec("//jaxlib/tools:build_gpu_plugin_wheel", "bazel-bin/jaxlib/tools/build_gpu_plugin_wheel"),
}

def add_python_argument(parser: argparse.ArgumentParser):
  """Add Python version argument to the parser."""
  parser.add_argument(
      "--python_version",
      type=str,
      choices=["3.10", "3.11", "3.12"],
      default="3.12",
      help="Python version to use",
  )

# Target system is assumed to be the host sytem (auto-detected) unless
# specified otherwise, e.g. for cross-compile builds
# allow override to pass in custom flags for certain builds like the RBE
# jobs
def add_system_argument(parser: argparse.ArgumentParser):
  """Add Target System argument to the parser."""
  parser.add_argument(
      "--target_system",
      type=str,
      default="",
      choices=["linux_x86_64", "linux_aarch64", "darwin_x86_64", "darwin_arm64", "windows_x86_64"],
      help="Target system to build for",
  )

def add_cuda_argument(parser: argparse.ArgumentParser):
  """Add CUDA version argument to the parser."""
  # TODO: should probably make this naming agnostic to allow for amd or intel
  parser.add_argument(
      "--cuda_version",
      type=str,
      default="12.3.2",
      help="CUDA version to use",
  )

def add_cudnn_argument(parser: argparse.ArgumentParser):
  """Add cuDNN version argument to the parser."""
  parser.add_argument(
      "--cudnn_version",
      type=str,
      default="9.1.1",
      help="cuDNN version to use",
  )

def add_rbe_argument(parser: argparse.ArgumentParser):
  """Add RBE mode to the parser."""
  parser.add_argument(
      "--use_rbe",
      type=bool,
      action="store_true",
      default=False,
      help="""
      If set, the build will use RBE where possible. Currently, only Linux x86
      and Windows builds can use RBE. On other platforms, setting this flag will
      be a no-op. RBE requires permissions to JAX's remote worker pool. Only
      Googlers and CI builds can use RBE.
      """,
  )

def add_clang_argument(parser: argparse.ArgumentParser):
  """Add Clang compiler argument to the parser."""
  parser.add_argument(
      "--use_clang",
      type=bool,
      action="store_true",
      default=False,
      help="""
      If set, the build will use Clang as the C++ compiler. Requires Clang to
      be present on the PATH or a path is given with --clang_path. CI builds use
      Clang by default.
      """,
  )

  parser.add_argument(
    "--clang_path",
    type=str,
    default="",
    help="""
    Path to the Clang binary to use. If not set and --use_clang is set, the
    build will attempt to find Clang on the PATH.
    """,
  )

def get_bazelrc_config(os_name: str, arch: str, artifact: str, mode:str, use_rbe: bool):
  """Returns the bazelrc config for the given architecture, OS, and build type."""
  bazelrc_config="{}_{}".format(os_name, arch)

  if mode == "local":
    # RBE is only supported on Linux x86 and Windows
    if use_rbe and (os_name == "linux" or os_name == "windows") and arch == "x86_64":
      bazelrc_config = "rbe_" + bazelrc_config
    else:
      bazelrc_config = "local_" + bazelrc_config
  else:
    # RBE is only supported on Linux x86 and Windows
    if (os_name == "linux" or os_name == "windows") and arch == "x86_64":
      bazelrc_config = "rbe_" + bazelrc_config
    else:
      bazelrc_config = "ci_" + bazelrc_config

  if artifact == "jax-cuda-plugin" or artifact == "jax-cuda-pjrt":
    bazelrc_config = bazelrc_config + "_cuda"

  return bazelrc_config

def get_jaxlib_git_hash():
  """Returns the git hash of the current repository."""
  res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
  return res.stdout

async def main():
  parser = argparse.ArgumentParser(
      description=(
          "JAX CLI for building/testing JAX, jaxlib, plugins, and pjrt."
      ),
  )

  parser.add_argument(
      "--mode",
      type=str,
      choices=["release", "local"],
      default="local",
      help="""
        Flags as requesting a release or release like build.  Setting this flag
        will assume multiple settings expected in release and CI builds. These
        are set by the release options in .bazelrc. To see best how this flag
        resolves you can run the artifact of choice with "--release -dry-run" to
        get the commands issued to Bazel for that artifact.
    """,
  )
  parser.add_argument(
      "--bazel_path",
      type=str,
      help=(
          "Path to the Bazel binary to use. The default is to find bazel via "
          "the PATH; if none is found, downloads a fresh copy of Bazelisk from "
          "GitHub."
      ),
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Prints the Bazel command that is going will be invoked.",
  )
  parser.add_argument("--verbose", action="store_true", help="Verbose output")

  global_args, remaining_args = parser.parse_known_args()

  # Create subparsers for jax, jaxlib, plugin, pjrt
  subparsers = parser.add_subparsers(
      dest="command", required=True, help="Artifact to build"
  )

  # JAX subcommand
  jax_parser = subparsers.add_parser("jax", help="Builds the JAX wheel package.")

  # Jaxlib subcommand
  jaxlib_parser = subparsers.add_parser("jaxlib", help="Builds the jaxlib package.")
  add_python_argument(jaxlib_parser)
  add_system_argument(jaxlib_parser)

  # jax-cuda-plugin subcommand
  plugin_parser = subparsers.add_parser("jax-cuda-plugin", help="Builds the jax-cuda-plugin package.")
  add_python_argument(plugin_parser)
  add_cuda_argument(plugin_parser)
  add_system_argument(plugin_parser)

  # jax-cuda-pjrt subcommand
  pjrt_parser = subparsers.add_parser("jax-cuda-pjrt", help="Builds the jax-cuda-pjrt package.")
  add_cuda_argument(pjrt_parser)
  add_system_argument(pjrt_parser)

  # Get the host systems architecture
  arch = platform.machine()
  # On Windows, this returns "amd64" instead of "x86_64. However, they both
  # are essentially the same.
  if arch.lower() == "amd64":
    arch = "x86_64"

  # Get the host system OS
  os_name = platform.system().lower()

  args = parser.parse_args(remaining_args)

  for key, value in vars(global_args).items():
    setattr(args, key, value)

  if args.command == "jax":
    logger.info("Building jax...")
  elif args.command == "jaxlib":
    logger.info(
        "Building jaxlib with python version %s for system %s",
        args.python_version,
        args.target_system,
    )
  elif args.command == "jax-cuda-plugin":
    logger.info("Building plugin...")
  elif args.command == "jax-cuda-pjrt":
    logger.info("Building pjrt...")
  else:
    logger.info("Invalid command")
    # print help and exit
    parser.print_help()
    sys.exit(1)

  # Find the path to Bazel
  bazel_path = tools.get_bazel_path(args.bazel_path)

  executor = command.SubprocessExecutor()

  bazel_command = command.CommandBuilder(bazel_path)
  # Temporary; when we make the new scripts as the default we can remove this.
  bazel_command.append("--bazelrc=ci/.bazelrc")
  bazel_command.append("build")

  bazel_command.append(
      "--config={}".format(get_bazelrc_config(os_name, arch, args.command, args.mode, args.use_rbe))
  )
  if hasattr(args, "python_version"):
    bazel_command.append(
        "--repo_env=HERMETIC_PYTHON_VERSION={}".format(args.python_version)
    )

  build_target, wheel_binary = ARTIFACT_BUILD_TARGET_DICT[args.command]
  bazel_command.append(build_target)

  logger.info("%s\n", bazel_command.command)

  if args.dry_run:
    logger.info("CLI is in dry run mode. Exiting without invoking Bazel.")
    sys.exit(0)

  await executor.run(bazel_command.command)

  logger.info("Building wheel...")
  run_wheel_binary = command.CommandBuilder(wheel_binary)

  output_dir = os.environ["JAXCI_OUTPUT_DIR"]
  run_wheel_binary.append("--output_path={}".format(output_dir))

  run_wheel_binary.append("--cpu={}".format(arch))

  if args.command == "jax-cuda-plugin" or args.command == "jax-cuda-pjrt":
    run_wheel_binary.append("--enable-cuda=True")
    major_cuda_version = args.cuda_version.split(".")[0]
    run_wheel_binary.append("--platform_version={}".format(major_cuda_version))

  jaxlib_git_hash = get_jaxlib_git_hash()
  run_wheel_binary.append("--jaxlib_git_hash={}".format(jaxlib_git_hash))

  logger.info("%s\n", run_wheel_binary.command)
  await executor.run(run_wheel_binary.command)

if __name__ == "__main__":
  asyncio.run(main())