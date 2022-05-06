from multiprocessing import Pool
from time import sleep
from kubernetes import client, config, utils
from query import *
import warnings

warnings.filterwarnings('ignore')
ts_namespace = 'train-ticket'

service_changes = [
    # normal changes
    ('ts-route-service', 'cqqcqq/route_inv_contacts:latest'),
    ('ts-order-service', 'cqqcqq/order_inv_contacts:latest'),
    ('ts-auth-service', 'cqqcqq/auth_inv_order:latest'),

    # abnormal changes
    ('ts-ticketinfo-service', 'cqqcqq/ticketinfo_oom:latest'),
    ('ts-travel-service', 'cqqcqq/travel_oom:latest'),

    ('ts-route-service', 'cqqcqq/route_sleep:latest'),
    ('ts-order-service', 'cqqcqq/order_sleep:latest'),
    ('ts-auth-service', 'cqqcqq/auth_sleep:latest'),

    ('ts-order-service', 'cqqcqq/order_port:latest'),
    ('ts-route-service', 'cqqcqq/route_port:latest'),
    ('ts-user-service', 'cqqcqq/user_port:latest'),

    ('ts-order-service', 'cqqcqq/order_table:latest'),
    ('ts-route-service', 'cqqcqq/route_table:latest'),
    ('ts-user-service', 'cqqcqq/user_table:latest')
]

query_func = {
    'ts-route-service': query_route,
    'ts-order-service': query_order,
    'ts-auth-service': query_auth,
    'ts-ticketinfo-service': query_ticketinfo,
    'ts-travel-service': query_travel,
    'ts-user-service': query_user,
}

change_order1 = [
    [0], [3], [6], [5], [7],
    [1], [7], [2], [0], [3],
    [11], [2], [5], [10], [8],
    [5], [2], [6], [5], [2],
    [1], [8], [9], [13], [11],
    [0], [1], [12], [0], [1],
    [2], [3], [1], [6], [9],
    [10], [4], [3], [8], [3],
    [11], [8], [0], [2], [8],
    [2], [0], [5], [6], [0],
]

change_order2 = [
    [3, 4], [10, 0], [12, 2], [3, 10], [0, 11],
    [9, 2], [5, 10], [2, 6], [9, 1], [6, 2],
    [8, 13], [13, 7], [8, 11], [11, 7], [2, 13],
    [0, 1], [10, 8], [2, 0], [0, 8], [1, 6],
    [12, 0], [0, 2], [8, 1], [1, 6], [2, 3],
    [0, 1], [10, 8], [12, 5], [3, 11], [2, 0],
    [1, 9], [2, 12], [6, 4], [0, 10], [0, 13],
    [11, 2], [1, 12], [7, 1], [1, 4], [13, 1],
    [2, 10], [1, 11], [6, 10], [2, 5], [1, 5],
    [4, 5], [8, 12], [0, 1], [13, 9], [0, 7],
]

change_order = change_order1


def wait_for_deployment_complete(api: client.AppsV1Api, name, timeout=300):
    start = time.time()

    while time.time()-start < timeout:
        time.sleep(10)
        response = api.read_namespaced_deployment_status(name, ts_namespace)
        s = response.status
        if (s.updated_replicas == response.spec.replicas and
                s.replicas == response.spec.replicas and
                s.available_replicas == response.spec.replicas and
                s.observed_generation >= response.metadata.generation):
            return True
        else:
            print(f'[INFO] [updated_replicas:{s.updated_replicas},replicas:{s.replicas}'
                  f',available_replicas:{s.available_replicas},observed_generation:{s.observed_generation}] waiting...')

    return True


def update_deployment_image(api: client.AppsV1Api, name, image) -> str:
    while True:
        deployment = api.read_namespaced_deployment(
            name=name, namespace=ts_namespace)
        old = deployment.spec.template.spec.containers[0].image
        # Update container image
        deployment.spec.template.spec.containers[0].image = image

        # patch the deployment
        try:
            resp = api.patch_namespaced_deployment(
                name=deployment.metadata.name, namespace=ts_namespace, body=deployment
            )
        except Exception:
            continue

        break

    print("[INFO] deployment's container image updated.\n")
    print("%s\t\t%s\t\t\t%s\t%s" %
          ("NAMESPACE", "NAME", "REVISION", "IMAGE"))
    print(
        "%s\t\t%s\t%s\t\t%s\n"
        % (
            resp.metadata.namespace,
            resp.metadata.name,
            resp.metadata.generation,
            resp.spec.template.spec.containers[0].image,
        )
    )

    return old


def main():
    config.load_kube_config()
    # k8s_client = client.ApiClient()
    api = client.AppsV1Api()
    request_period_log = []
    normal_change_log = []
    contact_image = update_deployment_image(
        api, 'ts-contacts-service', 'cqqcqq/contacts_sleep:latest')

    for order in change_order:
        old_images = []
        deploy_names = []
        wait_names = []
        print("-----------------------------------------")
        for order_id in order:
            # get current change
            change = service_changes[order_id]
            print("[INFO] curret deployment change:", change)
            # update deployment
            deploy_name = change[0]
            deploy_names.append(deploy_name)
            if order_id in [0, 1, 2, 5, 6, 7, 11, 12, 13]:
                # not port error service
                wait_names.append(deploy_name)
            new_image = change[1]

            old_image = update_deployment_image(
                api, deploy_name, new_image)
            old_images.append(old_image)

        # wait for completing update
        for name in wait_names:
            wait_for_deployment_complete(api, name)

        p = Pool(5)
        # send requests
        start = int(round(time.time() * 1000))
        if len(deploy_names) > 1:
            p.apply_async(query_func[deploy_names[1]])
            p.apply_async(query_func[deploy_names[1]])
        p.apply(query_func[deploy_names[0]])
        p.apply(query_func[deploy_names[0]])
        p.apply(query_func[deploy_names[0]])

        p.close()
        p.join()
        end = int(round(time.time() * 1000))

        root_services = []
        normal_services = []
        for i in order:
            if i < 3:
                normal_services.append(service_changes[i][0])
            else:
                root_services.append(service_changes[i][0])
        if len(normal_services) > 0:
            normal_change_log.append((normal_services, start, end))
        if len(root_services) > 0:
            request_period_log.append((root_services, start, end))
        # recover deployment
        for image in old_images:
            print(f"[INFO] recover deployment image to {image}")
            update_deployment_image(api, deploy_name, image)

        # wait 3 minutes
        print(f"[INFO] waitting...")
        sleep(180)

    print('-----------------------------------------')
    update_deployment_image(api, 'ts-contacts-service',
                            contact_image)
    print('request period log:')
    print(request_period_log)
    print('normal change log:')
    print(normal_change_log)
    print('End')


if __name__ == "__main__":
    main()

"""
{'api_version': 'apps/v1',
 'kind': 'Deployment',
 'metadata': {'annotations': {'deployment.kubernetes.io/revision': '1',
                              'kubectl.kubernetes.io/last-applied-configuration': '{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"name":"ts-travel-service","namespace":"train-ticket"},"spec":{"replicas":1,"selector":{"matchLabels":{"app":"ts-travel-service"}},"template":{"metadata":{"labels":{"app":"ts-travel-service"}},"spec":{"containers":[{"env":[{"name":"NODE_IP","valueFrom":{"fieldRef":{"fieldPath":"status.hostIP"}}},{"name":"SW_AGENT_COLLECTOR_BACKEND_SERVICES","value":"10.206.0.4:11800"},{"name":"SW_GRPC_LOG_SERVER_HOST","value":"10.206.0.4"},{"name":"SW_GRPC_LOG_SERVER_PORT","value":"11800"},{"name":"SW_AGENT_NAME","valueFrom":{"fieldRef":{"fieldPath":"metadata.labels[\'app\']"}}},{"name":"JAVA_TOOL_OPTIONS","value":"-javaagent:/skywalking/agent/skywalking-agent.jar"}],"image":"codewisdom/ts-travel-service:0.2.1","imagePullPolicy":"IfNotPresent","name":"ts-travel-service","ports":[{"containerPort":12346}],"readinessProbe":{"initialDelaySeconds":60,"periodSeconds":10,"tcpSocket":{"port":12346},"timeoutSeconds":5},"resources":{"limits":{"cpu":"1000m","memory":"1000Mi"},"requests":{"cpu":"300m","memory":"300Mi"}},"volumeMounts":[{"mountPath":"/skywalking","name":"skywalking-agent"}]}],"initContainers":[{"args":["-c","cp '
                                                                                  '-R '
                                                                                  '/skywalking/agent '
                                                                                  '/agent/"],"command":["/bin/sh"],"image":"apache/skywalking-java-agent:8.8.0-alpine","name":"agent-container","volumeMounts":[{"mountPath":"/agent","name":"skywalking-agent"}]}],"volumes":[{"emptyDir":{},"name":"skywalking-agent"}]}}}}\n'},
              'cluster_name': None,
              'creation_timestamp': datetime.datetime(2022, 3, 13, 13, 26, 51, tzinfo=tzutc()),
              'deletion_grace_period_seconds': None,
              'deletion_timestamp': None,
              'finalizers': None,
              'generate_name': None,
              'generation': 1,
              'labels': None,
              'managed_fields': [{'api_version': 'apps/v1',
                                  'fields_type': 'FieldsV1',
                                  'fields_v1': {'f:metadata': {'f:annotations': {'.': {},
                                                                                 'f:kubectl.kubernetes.io/last-applied-configuration': {}}},
                                                'f:spec': {'f:progressDeadlineSeconds': {},
                                                           'f:replicas': {},
                                                           'f:revisionHistoryLimit': {},
                                                           'f:selector': {},
                                                           'f:strategy': {'f:rollingUpdate': {'.': {},
                                                                                              'f:maxSurge': {},
                                                                                              'f:maxUnavailable': {}},
                                                                          'f:type': {}},
                                                           'f:template': {'f:metadata': {'f:labels': {'.': {},
                                                                                                      'f:app': {}}},
                                                                          'f:spec': {'f:containers': {'k:{"name":"ts-travel-service"}': {'.': {},
                                                                                                                                         'f:env': {'.': {},
                                                                                                                                                   'k:{"name":"JAVA_TOOL_OPTIONS"}': {'.': {},
                                                                                                                                                                                      'f:name': {},
                                                                                                                                                                                      'f:value': {}},
                                                                                                                                                   'k:{"name":"NODE_IP"}': {'.': {},
                                                                                                                                                                            'f:name': {},
                                                                                                                                                                            'f:valueFrom': {'.': {},
                                                                                                                                                                                            'f:fieldRef': {}}},
                                                                                                                                                   'k:{"name":"SW_AGENT_COLLECTOR_BACKEND_SERVICES"}': {'.': {},
                                                                                                                                                                                                        'f:name': {},
                                                                                                                                                                                                        'f:value': {}},
                                                                                                                                                   'k:{"name":"SW_AGENT_NAME"}': {'.': {},
                                                                                                                                                                                  'f:name': {},
                                                                                                                                                                                  'f:valueFrom': {'.': {},
                                                                                                                                                                                                  'f:fieldRef': {}}},
                                                                                                                                                   'k:{"name":"SW_GRPC_LOG_SERVER_HOST"}': {'.': {},
                                                                                                                                                                                            'f:name': {},
                                                                                                                                                                                            'f:value': {}},
                                                                                                                                                   'k:{"name":"SW_GRPC_LOG_SERVER_PORT"}': {'.': {},
                                                                                                                                                                                            'f:name': {},
                                                                                                                                                                                            'f:value': {}}},
                                                                                                                                         'f:image': {},
                                                                                                                                         'f:imagePullPolicy': {},
                                                                                                                                         'f:name': {},
                                                                                                                                         'f:ports': {'.': {},
                                                                                                                                                     'k:{"containerPort":12346,"protocol":"TCP"}': {'.': {},
                                                                                                                                                                                                    'f:containerPort': {},
                                                                                                                                                                                                    'f:protocol': {}}},
                                                                                                                                         'f:readinessProbe': {'.': {},
                                                                                                                                                              'f:failureThreshold': {},
                                                                                                                                                              'f:initialDelaySeconds': {},
                                                                                                                                                              'f:periodSeconds': {},
                                                                                                                                                              'f:successThreshold': {},
                                                                                                                                                              'f:tcpSocket': {'.': {},
                                                                                                                                                                              'f:port': {}},
                                                                                                                                                              'f:timeoutSeconds': {}},
                                                                                                                                         'f:resources': {'.': {},
                                                                                                                                                         'f:limits': {'.': {},
                                                                                                                                                                      'f:cpu': {},
                                                                                                                                                                      'f:memory': {}},
                                                                                                                                                         'f:requests': {'.': {},
                                                                                                                                                                        'f:cpu': {},
                                                                                                                                                                        'f:memory': {}}},
                                                                                                                                         'f:terminationMessagePath': {},
                                                                                                                                         'f:terminationMessagePolicy': {},
                                                                                                                                         'f:volumeMounts': {'.': {},
                                                                                                                                                            'k:{"mountPath":"/skywalking"}': {'.': {},
                                                                                                                                                                                              'f:mountPath': {},
                                                                                                                                                                                              'f:name': {}}}}},
                                                                                     'f:dnsPolicy': {},
                                                                                     'f:initContainers': {'.': {},
                                                                                                          'k:{"name":"agent-container"}': {'.': {},
                                                                                                                                           'f:args': {},
                                                                                                                                           'f:command': {},
                                                                                                                                           'f:image': {},
                                                                                                                                           'f:imagePullPolicy': {},
                                                                                                                                           'f:name': {},
                                                                                                                                           'f:resources': {},
                                                                                                                                           'f:terminationMessagePath': {},
                                                                                                                                           'f:terminationMessagePolicy': {},
                                                                                                                                           'f:volumeMounts': {'.': {},
                                                                                                                                                              'k:{"mountPath":"/agent"}': {'.': {},
                                                                                                                                                                                           'f:mountPath': {},
                                                                                                                                                                                           'f:name': {}}}}},
                                                                                     'f:restartPolicy': {},
                                                                                     'f:schedulerName': {},
                                                                                     'f:securityContext': {},
                                                                                     'f:terminationGracePeriodSeconds': {},
                                                                                     'f:volumes': {'.': {},
                                                                                                   'k:{"name":"skywalking-agent"}': {'.': {},
                                                                                                                                     'f:emptyDir': {},
                                                                                                                                     'f:name': {}}}}}}},
                                  'manager': 'kubectl-client-side-apply',
                                  'operation': 'Update',
                                  'subresource': None,
                                  'time': datetime.datetime(2022, 3, 13, 13, 26, 51, tzinfo=tzutc())},
                                 {'api_version': 'apps/v1',
                                  'fields_type': 'FieldsV1',
                                  'fields_v1': {'f:metadata': {'f:annotations': {'f:deployment.kubernetes.io/revision': {}}},
                                                'f:status': {'f:availableReplicas': {},
                                                             'f:conditions': {'.': {},
                                                                              'k:{"type":"Available"}': {'.': {},
                                                                                                         'f:lastTransitionTime': {},
                                                                                                         'f:lastUpdateTime': {},
                                                                                                         'f:message': {},
                                                                                                         'f:reason': {},
                                                                                                         'f:status': {},
                                                                                                         'f:type': {}},
                                                                              'k:{"type":"Progressing"}': {'.': {},
                                                                                                           'f:lastTransitionTime': {},
                                                                                                           'f:lastUpdateTime': {},
                                                                                                           'f:message': {},
                                                                                                           'f:reason': {},
                                                                                                           'f:status': {},
                                                                                                           'f:type': {}}},
                                                             'f:observedGeneration': {},
                                                             'f:readyReplicas': {},
                                                             'f:replicas': {},
                                                             'f:updatedReplicas': {}}},
                                  'manager': 'kube-controller-manager',
                                  'operation': 'Update',
                                  'subresource': 'status',
                                  'time': datetime.datetime(2022, 3, 13, 13, 29, 31, tzinfo=tzutc())}],
              'name': 'ts-travel-service',
              'namespace': 'train-ticket',
              'owner_references': None,
              'resource_version': '18431892',
              'self_link': None,
              'uid': '94220a73-8084-4c90-ae73-3fe83ca54db3'},
 'spec': {'min_ready_seconds': None,
          'paused': None,
          'progress_deadline_seconds': 600,
          'replicas': 1,
          'revision_history_limit': 10,
          'selector': {'match_expressions': None,
                       'match_labels': {'app': 'ts-travel-service'}},
          'strategy': {'rolling_update': {'max_surge': '25%',
                                          'max_unavailable': '25%'},
                       'type': 'RollingUpdate'},
          'template': {'metadata': {'annotations': None,
                                    'cluster_name': None,
                                    'creation_timestamp': None,
                                    'deletion_grace_period_seconds': None,
                                    'deletion_timestamp': None,
                                    'finalizers': None,
                                    'generate_name': None,
                                    'generation': None,
                                    'labels': {'app': 'ts-travel-service'},
                                    'managed_fields': None,
                                    'name': None,
                                    'namespace': None,
                                    'owner_references': None,
                                    'resource_version': None,
                                    'self_link': None,
                                    'uid': None},
                       'spec': {'active_deadline_seconds': None,
                                'affinity': None,
                                'automount_service_account_token': None,
                                'containers': [{'args': None,
                                                'command': None,
                                                'env': [{'name': 'NODE_IP',
                                                         'value': None,
                                                         'value_from': {'config_map_key_ref': None,
                                                                        'field_ref': {'api_version': 'v1',
                                                                                      'field_path': 'status.hostIP'},
                                                                        'resource_field_ref': None,
                                                                        'secret_key_ref': None}},
                                                        {'name': 'SW_AGENT_COLLECTOR_BACKEND_SERVICES',
                                                         'value': '10.206.0.4:11800',
                                                         'value_from': None},
                                                        {'name': 'SW_GRPC_LOG_SERVER_HOST',
                                                         'value': '10.206.0.4',
                                                         'value_from': None},
                                                        {'name': 'SW_GRPC_LOG_SERVER_PORT',
                                                         'value': '11800',
                                                         'value_from': None},
                                                        {'name': 'SW_AGENT_NAME',
                                                         'value': None,
                                                         'value_from': {'config_map_key_ref': None,
                                                                        'field_ref': {'api_version': 'v1',
                                                                                      'field_path': "metadata.labels['app']"},
                                                                        'resource_field_ref': None,
                                                                        'secret_key_ref': None}},
                                                        {'name': 'JAVA_TOOL_OPTIONS',
                                                         'value': '-javaagent:/skywalking/agent/skywalking-agent.jar',
                                                         'value_from': None}],
                                                'env_from': None,
                                                'image': 'codewisdom/ts-travel-service:0.2.1',
                                                'image_pull_policy': 'IfNotPresent',
                                                'lifecycle': None,
                                                'liveness_probe': None,
                                                'name': 'ts-travel-service',
                                                'ports': [{'container_port': 12346,
                                                           'host_ip': None,
                                                           'host_port': None,
                                                           'name': None,
                                                           'protocol': 'TCP'}],
                                                'readiness_probe': {'_exec': None,
                                                                    'failure_threshold': 3,
                                                                    'grpc': None,
                                                                    'http_get': None,
                                                                    'initial_delay_seconds': 60,
                                                                    'period_seconds': 10,
                                                                    'success_threshold': 1,
                                                                    'tcp_socket': {'host': None,
                                                                                   'port': 12346},
                                                                    'termination_grace_period_seconds': None,
                                                                    'timeout_seconds': 5},
                                                'resources': {'limits': {'cpu': '1',
                                                                         'memory': '1000Mi'},
                                                              'requests': {'cpu': '300m',
                                                                           'memory': '300Mi'}},
                                                'security_context': None,
                                                'startup_probe': None,
                                                'stdin': None,
                                                'stdin_once': None,
                                                'termination_message_path': '/dev/termination-log',
                                                'termination_message_policy': 'File',
                                                'tty': None,
                                                'volume_devices': None,
                                                'volume_mounts': [{'mount_path': '/skywalking',
                                                                   'mount_propagation': None,
                                                                   'name': 'skywalking-agent',
                                                                   'read_only': None,
                                                                   'sub_path': None,
                                                                   'sub_path_expr': None}],
                                                'working_dir': None}],
                                'dns_config': None,
                                'dns_policy': 'ClusterFirst',
                                'enable_service_links': None,
                                'ephemeral_containers': None,
                                'host_aliases': None,
                                'host_ipc': None,
                                'host_network': None,
                                'host_pid': None,
                                'hostname': None,
                                'image_pull_secrets': None,
                                'init_containers': [{'args': ['-c',
                                                              'cp -R '
                                                              '/skywalking/agent '
                                                              '/agent/'],
                                                     'command': ['/bin/sh'],
                                                     'env': None,
                                                     'env_from': None,
                                                     'image': 'apache/skywalking-java-agent:8.8.0-alpine',
                                                     'image_pull_policy': 'IfNotPresent',
                                                     'lifecycle': None,
                                                     'liveness_probe': None,
                                                     'name': 'agent-container',
                                                     'ports': None,
                                                     'readiness_probe': None,
                                                     'resources': {'limits': None,
                                                                   'requests': None},
                                                     'security_context': None,
                                                     'startup_probe': None,
                                                     'stdin': None,
                                                     'stdin_once': None,
                                                     'termination_message_path': '/dev/termination-log',
                                                     'termination_message_policy': 'File',
                                                     'tty': None,
                                                     'volume_devices': None,
                                                     'volume_mounts': [{'mount_path': '/agent',
                                                                        'mount_propagation': None,
                                                                        'name': 'skywalking-agent',
                                                                        'read_only': None,
                                                                        'sub_path': None,
                                                                        'sub_path_expr': None}],
                                                     'working_dir': None}],
                                'node_name': None,
                                'node_selector': None,
                                'os': None,
                                'overhead': None,
                                'preemption_policy': None,
                                'priority': None,
                                'priority_class_name': None,
                                'readiness_gates': None,
                                'restart_policy': 'Always',
                                'runtime_class_name': None,
                                'scheduler_name': 'default-scheduler',
                                'security_context': {'fs_group': None,
                                                     'fs_group_change_policy': None,
                                                     'run_as_group': None,
                                                     'run_as_non_root': None,
                                                     'run_as_user': None,
                                                     'se_linux_options': None,
                                                     'seccomp_profile': None,
                                                     'supplemental_groups': None,
                                                     'sysctls': None,
                                                     'windows_options': None},
                                'service_account': None,
                                'service_account_name': None,
                                'set_hostname_as_fqdn': None,
                                'share_process_namespace': None,
                                'subdomain': None,
                                'termination_grace_period_seconds': 30,
                                'tolerations': None,
                                'topology_spread_constraints': None,
                                'volumes': [{'aws_elastic_block_store': None,
                                             'azure_disk': None,
                                             'azure_file': None,
                                             'cephfs': None,
                                             'cinder': None,
                                             'config_map': None,
                                             'csi': None,
                                             'downward_api': None,
                                             'empty_dir': {'medium': None,
                                                           'size_limit': None},
                                             'ephemeral': None,
                                             'fc': None,
                                             'flex_volume': None,
                                             'flocker': None,
                                             'gce_persistent_disk': None,
                                             'git_repo': None,
                                             'glusterfs': None,
                                             'host_path': None,
                                             'iscsi': None,
                                             'name': 'skywalking-agent',
                                             'nfs': None,
                                             'persistent_volume_claim': None,
                                             'photon_persistent_disk': None,
                                             'portworx_volume': None,
                                             'projected': None,
                                             'quobyte': None,
                                             'rbd': None,
                                             'scale_io': None,
                                             'secret': None,
                                             'storageos': None,
                                             'vsphere_volume': None}]}}},
 'status': {'available_replicas': 1,
            'collision_count': None,
            'conditions': [{'last_transition_time': datetime.datetime(2022, 3, 13, 13, 29, 31, tzinfo=tzutc()),
                            'last_update_time': datetime.datetime(2022, 3, 13, 13, 29, 31, tzinfo=tzutc()),
                            'message': 'Deployment has minimum availability.',
                            'reason': 'MinimumReplicasAvailable',
                            'status': 'True',
                            'type': 'Available'},
                           {'last_transition_time': datetime.datetime(2022, 3, 13, 13, 27, tzinfo=tzutc()),
                            'last_update_time': datetime.datetime(2022, 3, 13, 13, 29, 31, tzinfo=tzutc()),
                            'message': 'ReplicaSet '
                                       '"ts-travel-service-594b6bc756" has '
                                       'successfully progressed.',
                            'reason': 'NewReplicaSetAvailable',
                            'status': 'True',
                            'type': 'Progressing'}],
            'observed_generation': 1,
            'ready_replicas': 1,
            'replicas': 1,
            'unavailable_replicas': None,
            'updated_replicas': 1}}
"""
