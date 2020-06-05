---
title: AKS deployment guide
sidebar_label: AKS deployment
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

This document outlines the installation steps for Azure Kubernetes Service (AKS).

## Launch an AKS cluster
:::important
Make sure [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) (`az`) is installed before proceeding.
:::

We recommend launching a cluster with 2 `Standard_D4s_v3` nodes to start, with autoscaling and network policy enabled. You can add additional CPU/GPU node pools as needed later.

Here is a sample `az` command to create a bare minimum cluster:

```bash
az aks create --resource-group <resource-group> --name <cluster-name> \
    --node-count 2 \
    --node-vm-size Standard_D4s_v3 \
    --node-osdisk-size 100 \
    --min-count 2 \
    --max-count 2 \
    --enable-cluster-autoscaler \
    --network-plugin azure \
    --network-policy azure \
    --enable-addons monitoring \
    --generate-ssh-keys
```

:::note
The `--enable-addons monitoring` flag in the command above enables Azure Monitor for log aggregation which can incur additional charges. You can optionally remove this flag and add `--enable-efk-logging` to `opctl` command below.
:::

:::note
You can specify the version of the cluster.
Get a list of versions by running:
```shell script
az aks get-versions --location eastus --output table
```
Example output
```text
KubernetesVersion    Upgrades
-------------------  -------------------------------------------------
1.18.2(preview)      None available
1.18.1(preview)      1.18.2(preview)
1.17.5(preview)      1.18.1(preview), 1.18.2(preview)
1.17.4(preview)      1.17.5(preview), 1.18.1(preview), 1.18.2(preview)
1.16.9               1.17.4(preview), 1.17.5(preview)
1.16.8               1.16.9, 1.17.4(preview), 1.17.5(preview)
1.15.11              1.16.8, 1.16.9
1.15.10              1.15.11, 1.16.8, 1.16.9
1.14.8               1.15.10, 1.15.11
1.14.7               1.14.8, 1.15.10, 1.15.11
```

Add the flag to the above command:
```shell script
az aks create --resource-group <resource-group> --name <cluster-name> \
    --node-count 2 \
    --kubernetes-version 1.16.9 \
    ...
```

:::

You can then get access credentials by running:

```
az aks get-credentials --resource-group <resource-group> --name <cluster-name> --admin
```

## Install Onepanel
1. Download the latest `opctl` for your operating system from [our release page](https://github.com/onepanelio/core/releases/latest).

<Tabs
  defaultValue="linux"
  values={[
    { label: 'Linux', value: 'linux', },
    { label: 'macOS', value: 'macos', },
  ]
}>
<TabItem value="linux">

```bash
# Download the binary
curl -sLO https://github.com/onepanelio/core/releases/download/latest/opctl-linux-amd64

# Make binary executable
chmod +x opctl-linux-amd64

# Move binary to path
mv ./opctl-linux-amd64 /usr/local/bin/opctl

# Test installation
opctl version
```

</TabItem>
<TabItem value="macos">

```bash
# Download the binary
curl -sLO https://github.com/onepanelio/core/releases/download/latest/opctl-macos-amd64

# Make binary executable
chmod +x opctl-macos-amd64

# Move binary to path
mv ./opctl-macos-amd64 /usr/local/bin/opctl

# Test installation
opctl version
```

</TabItem>
</Tabs>

2. Run the following command to initialize a `params.yaml` template for AKS:

```bash
opctl init --provider aks \
    --enable-https \
    --enable-cert-manager \
    --dns-provider <dns-provider>
```

:::note
The `--enable-https` flag is optional and requires a TLS certificate, but it is highly recommended. You can optionally set the `--enable-cert-manager` and `--dns-provider` flags, so TLS certificates are automatically created and renewed via [Let's Encrypt](https://letsencrypt.org/). If you do not set this flag and your DNS provider isn't one of the [supported DNS providers](/docs/deployment/configuration/tls#supported-dns-providers), then you have to create a wildcard certificate and manually manage it.
:::

:::note
If you have GPU nodes, you need to set the `--gpu-device-plugins` flag. Valid values are `nvidia` and `amd` or a comma separated combination of both `nvidia,amd`.
:::

3. Populate `params.yaml` by following the instructions in the template, you can also refer to [configuration files](/docs/deployment/configuration/files) for more detailed information.

4. Finally, run the following command to deploy to your cluster:

```bash
opctl apply
```

:::note
If the command completes but it indicates that your cluster is not ready, you can check status again by running `opctl app status`. If you're still seeing issues, run `kubectl get pods --all-namespaces` to see if there are any crashing pods.
:::

5. Once the deployment completes, the CLI will display the IP and wildcard domain you need to use to setup your DNS. You can also get this information again by running:

```bash
opctl app status
```

6. Create an `A` record in your DNS provider based on the instructions above.

:::tip
Note that you should use a wildcard `A` record, for example: `*.example.com` or `*.subdomain.example.com`
:::

7. Use the following command to get your auth token to log into Onepanel:

```bash
opctl auth token
```
