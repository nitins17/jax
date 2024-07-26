package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

platform(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/ml-devinfra-linux-aarch64-cross-compile@sha256:3760fac1156dffd2b5d0469c4e6c2fc436288a7c5205b84492c27daf8aaa2893",
        "OSFamily": "Linux",
    },
)

platform(
    name = "darwin_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

platform(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

platform(
    name = "darwin_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

toolchain(
    name = "macos-x86-cross-compile-cc-toolchain",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    toolchain = "//tools/toolchains/cross_compile/cc:macos_x86_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

toolchain(
    name = "macos-arm64-cross-compile-cc-toolchain",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
    toolchain = "//tools/toolchains/cross_compile/cc:macos_arm64_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)
