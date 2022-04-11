from kubernetes import client, config, utils

update_svc_num = 1
total_loop = 50

normal_order = [
    ('route', 'image:tag'),
    ('order', ''),
    ('auth', ''),
]

abnormal_order = [
    ('ticket-info', ''),
    ('trave', ''),
    ('route', ''),
    ('order', ''),
    ('auth', ''),
    ('user', ''),
]


def update_deployment(api, deployment):
    # Update container image
    deployment.spec.template.spec.containers[0].image = "nginx:1.16.0"

    # patch the deployment
    resp = api.patch_namespaced_deployment(
        name='', namespace="default", body=deployment
    )

    print("\n[INFO] deployment's container image updated.\n")
    print("%s\t%s\t\t\t%s\t%s" % ("NAMESPACE", "NAME", "REVISION", "IMAGE"))
    print(
        "%s\t\t%s\t%s\t\t%s\n"
        % (
            resp.metadata.namespace,
            resp.metadata.name,
            resp.metadata.generation,
            resp.spec.template.spec.containers[0].image,
        )
    )


def main():
    config.load_kube_config()
    k8s_client = client.ApiClient()
    yaml_file = 'examples/configmap-demo-pod.yml'
    utils.create_from_yaml(k8s_client, yaml_file, verbose=True)


if __name__ == "__main__":
    main()
