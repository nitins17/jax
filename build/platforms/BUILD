# //build:platforms/BUILD

load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "BASE_JDK9_JVM_OPTS")

package(default_visibility = ["//visibility:public"])

java_runtime(
    name = "local_jdk",
    java_home = "/Library/Java/JavaVirtualMachines/default/Contents/Home",
    tags = ["manual"],
)

java_toolchain(
    name = "local_jdk_config",
    genclass = "@bazel_tools//tools/jdk:genclass",
    ijar = "@bazel_tools//tools/jdk:ijar",
    java_runtime = ":local_jdk",
    javabuilder = "@bazel_tools//tools/jdk:javabuilder",
    jvm_opts = BASE_JDK9_JVM_OPTS,
    singlejar = "@bazel_tools//tools/jdk:singlejar",
)

toolchain(
    name = "local_jdk_toolchain",
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
    toolchain = "local_jdk_config",
    toolchain_type = "@bazel_tools//tools/jdk:toolchain_type",
)



platform(
    name = "rbe_linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    remote_execution_properties = """
        properties: {
            name: "Pool"
            value: "linux-x86-cpu-pool"
        }
         properties: {
             name: "container-image"
             value: "docker://us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build@sha256:ea67e8453d8b09c2ba48853da5e79efef4b65804b4a48dfae4b4da89ffd38405"
         }
    """
)

platform(
    name = "rbe_linux_arm64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:arm64",
    ],
    remote_execution_properties = """
        properties: {
            name: "Pool"
            value: "linux-arm64-pool"
        }
        # properties: {
        #     name: "container-image"
        #     value: "docker://..." 
        # }
    """
)

platform(
    name = "rbe_macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
exec_properties = {
        "OSFamily": "Darwin",
        "OSArch": "arm64",
        "Pool": "mac-silicon-pool",
        "sandbox": "none",
    },
#    remote_execution_properties = """
#       properties: {
#            name: "Pool"
#            value: "mac-silicon-pool"
#        }
#    """
)
