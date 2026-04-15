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
        # Optionally, specify the container image if not using the default
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
        # You'll likely need a specific Arm64 container image
        # properties: {
        #     name: "container-image"
        #     value: "docker://..." # Path to your Arm64 toolchain container
        # }
    """
)

