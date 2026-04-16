# //build:platforms/BUILD

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
    remote_execution_properties = """
        properties: {
            name: "Pool"
            value: "mac-silicon-pool"
        }
    """
)
