/*! For license information please see fbd7a87c.8a27d1a3.js.LICENSE.txt */
(window.webpackJsonp=window.webpackJsonp||[]).push([[44],{145:function(e,a,t){"use strict";t.r(a),t.d(a,"frontMatter",(function(){return r})),t.d(a,"metadata",(function(){return l})),t.d(a,"rightToc",(function(){return b})),t.d(a,"default",(function(){return d}));var n=t(2),c=(t(0),t(149)),o=t(156),i=t(154);const r={title:"Quick start",sidebar_label:"Quick start"},l={id:"getting-started/quickstart",title:"Quick start",description:"It's easy to get started with Onepanel. First, install the CLI (opctl) and then using opctl, generate a params.yaml file and configure your deployment. Once complete, you can access your deployment from any browser, use your Kubernetes auth to login and finally run a workflow.",source:"@site/docs/getting-started/quickstart.md",permalink:"/docs/getting-started/quickstart",editUrl:"https://github.com/onepanelio/core-docs/tree/master/docs/getting-started/quickstart.md",sidebar_label:"Quick start",sidebar:"gettingStarted",next:{title:"Namespaces",permalink:"/docs/getting-started/concepts/namespaces"}},b=[{value:"Step 0: Create a Kubernetes cluster",id:"step-0-create-a-kubernetes-cluster",children:[]},{value:"Step 1: Install Onepanel",id:"step-1-install-onepanel",children:[]}],s={rightToc:b};function d({components:e,...a}){return Object(c.b)("wrapper",Object(n.a)({},s,a,{components:e,mdxType:"MDXLayout"}),Object(c.b)("p",null,"It's easy to get started with Onepanel. First, install the CLI (",Object(c.b)("inlineCode",{parentName:"p"},"opctl"),") and then using ",Object(c.b)("inlineCode",{parentName:"p"},"opctl"),", generate a ",Object(c.b)("inlineCode",{parentName:"p"},"params.yaml")," file and configure your deployment. Once complete, you can access your deployment from any browser, use your Kubernetes auth to login and finally run a workflow."),Object(c.b)("h2",{id:"step-0-create-a-kubernetes-cluster"},"Step 0: Create a Kubernetes cluster"),Object(c.b)("p",null,"First, create a Kubernetes cluster in one of the following cloud providers:"),Object(c.b)(o.a,{defaultValue:"aks",values:[{label:"Azure AKS",value:"aks"},{label:"Amazon EKS",value:"eks"},{label:"Google Cloud GKE",value:"gke"}],mdxType:"Tabs"},Object(c.b)(i.a,{value:"aks",mdxType:"TabItem"},Object(c.b)("div",{className:"admonition admonition-important alert alert--info"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"important")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"Make sure ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest"}),"Azure CLI")," (",Object(c.b)("inlineCode",{parentName:"p"},"az"),") is installed before proceeding."))),Object(c.b)("p",null,"We recommend launching a cluster with 2 ",Object(c.b)("inlineCode",{parentName:"p"},"Standard_D4s_v3")," nodes to start, with autoscaling and network policy enabled. You can add additional CPU/GPU node pools as needed later."),Object(c.b)("p",null,"Here is a sample ",Object(c.b)("inlineCode",{parentName:"p"},"az")," command to create a bare minimum cluster:"),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"az aks create --resource-group <resource-group> --name <cluster-name> \\\n    --node-count 2 \\\n    --node-vm-size Standard_D4s_v3 \\\n    --node-osdisk-size 100 \\\n    --min-count 2 \\\n    --max-count 2 \\\n    --enable-cluster-autoscaler \\\n    --network-plugin azure \\\n    --network-policy azure \\\n    --enable-addons monitoring \\\n    --generate-ssh-keys\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"The ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-addons monitoring")," flag in the command above enables Azure Monitor for log aggregation which can incur additional charges. You can optionally remove this flag and add ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-efk-logging")," to ",Object(c.b)("inlineCode",{parentName:"p"},"opctl")," command below."))),Object(c.b)("p",null,"You can then get access credentials by running:"),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{}),"az aks get-credentials --resource-group <resource-group> --name <cluster-name> --admin\n"))),Object(c.b)(i.a,{value:"eks",mdxType:"TabItem"},Object(c.b)("div",{className:"admonition admonition-important alert alert--info"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"important")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"Make sure ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://eksctl.io/introduction/#installation"}),"Amazon EKS CLI")," (",Object(c.b)("inlineCode",{parentName:"p"},"eksctl"),") is installed before proceeding."))),Object(c.b)("p",null,"We recommend launching a cluster with 2 ",Object(c.b)("inlineCode",{parentName:"p"},"m5.xlarge")," nodes to start, with autoscaling and network policy enabled. You can add additional CPU/GPU node pools as needed later."),Object(c.b)("p",null,"Here are sample ",Object(c.b)("inlineCode",{parentName:"p"},"eksctl")," commands to create a bare minimum cluster:"),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"eksctl create cluster --name=<cluster-name> --region <region> \\\n    --nodes 2  \\\n    --node-volume-size 100 \\\n    --nodes-min 2 \\\n    --nodes-max 2 \\\n    --asg-access \\\n    --managed \\\n    --ssh-access\n")),Object(c.b)("p",null,"To enable auto scaling see ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://eksctl.io/usage/autoscaling/"}),"Enable Auto Scaling")),Object(c.b)("p",null,"To enable network policy see ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://docs.aws.amazon.com/eks/latest/userguide/calico.html"}),"Installing Calico on Amazon EKS")),Object(c.b)("p",null,"To enable logging see ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://eksctl.io/usage/cloudwatch-cluster-logging/"}),"Enabling CloudWatch logging")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"You can optionally skip the logging configuration above and add ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-efk-logging")," to ",Object(c.b)("inlineCode",{parentName:"p"},"opctl")," command below."))),Object(c.b)("p",null,"The ",Object(c.b)("inlineCode",{parentName:"p"},"eksctl")," command above will automatically retrieve your cluster's access credentials but you can also get them by running:"),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{}),"eksctl utils write-kubeconfig --cluster=<cluster-name> --region <region>\n"))),Object(c.b)(i.a,{value:"gke",mdxType:"TabItem"},Object(c.b)("div",{className:"admonition admonition-important alert alert--info"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"important")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"Make sure ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://cloud.google.com/sdk/install"}),"Google Cloud SDK")," (",Object(c.b)("inlineCode",{parentName:"p"},"gcloud"),") is installed before proceeding."))),Object(c.b)("p",null,"We recommend launching a cluster with 2 ",Object(c.b)("inlineCode",{parentName:"p"},"n1-standard-4")," nodes to start, with autoscaling and network policy enabled. You can add additional CPU/GPU node pools as needed later."),Object(c.b)("p",null,"Here is sample ",Object(c.b)("inlineCode",{parentName:"p"},"gcloud")," command to create a bare minimum cluster:"),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"gcloud container --project <project-name> clusters create <cluster-name> --zone <zone> \\\n    --num-nodes 2 \\\n    --machine-type n1-standard-4 \\\n    --disk-size 100 \\\n    --min-nodes 0 \\\n    --max-nodes 2 \\\n    --enable-autoscaling \\\n    --enable-network-policy \\\n    --enable-stackdriver-kubernetes \\\n    --addons HorizontalPodAutoscaling,HttpLoadBalancing\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"The ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-stackdriver-kubernetes")," flag in above command enables Google Stackdriver for log aggregation which can incur additional charges. You can optionally remove this flag and add ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-efk-logging")," to ",Object(c.b)("inlineCode",{parentName:"p"},"opctl")," command below."))),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"You can optionally add the ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-tpu")," flag to enable TPUs in GKE."))),Object(c.b)("p",null,"The command above will automatically retrieve your cluster's access credentials but you can also get them by running:"),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{}),"gcloud container clusters get-credentials <cluster-name> --zone <zone>\n")))),Object(c.b)("h2",{id:"step-1-install-onepanel"},"Step 1: Install Onepanel"),Object(c.b)("ol",null,Object(c.b)("li",{parentName:"ol"},"Download the latest ",Object(c.b)("inlineCode",{parentName:"li"},"opctl")," for your operating system from ",Object(c.b)("a",Object(n.a)({parentName:"li"},{href:"https://github.com/onepanelio/core/releases/latest"}),"our release page"),".")),Object(c.b)(o.a,{defaultValue:"linux",values:[{label:"Linux",value:"linux"},{label:"macOS",value:"macos"}],mdxType:"Tabs"},Object(c.b)(i.a,{value:"linux",mdxType:"TabItem"},Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"# Download the binary\ncurl -sLO https://github.com/onepanelio/core/releases/download/v0.9.0/opctl-linux-amd64\n\n# Make binary executable\nchmod +x opctl-linux-amd64\n\n# Move binary to path\nmv ./opctl-linux-amd64 /usr/local/bin/opctl\n\n# Test installation\nopctl version\n"))),Object(c.b)(i.a,{value:"macos",mdxType:"TabItem"},Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"# Download the binary\ncurl -sLO https://github.com/onepanelio/core/releases/download/v0.9.0/opctl-macos-amd64\n\n# Make binary executable\nchmod +x opctl-macos-amd64\n\n# Move binary to path\nmv ./opctl-macos-amd64 /usr/local/bin/opctl\n\n# Test installation\nopctl version\n")))),Object(c.b)("ol",{start:2},Object(c.b)("li",{parentName:"ol"},"Run the following command to initialize a ",Object(c.b)("inlineCode",{parentName:"li"},"params.yaml")," template for your provider:")),Object(c.b)(o.a,{defaultValue:"aks",values:[{label:"Azure AKS",value:"aks"},{label:"Amazon EKS",value:"eks"},{label:"Google Cloud GKE",value:"gke"}],mdxType:"Tabs"},Object(c.b)(i.a,{value:"aks",mdxType:"TabItem"},Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl init --provider aks \\\n  --enable-https \\\n  --enable-cert-manager \\\n  --dns-provider <dns-provider>\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"If you have GPU nodes, you need to set the ",Object(c.b)("inlineCode",{parentName:"p"},"--gpu-device-plugins")," flag. Valid values are ",Object(c.b)("inlineCode",{parentName:"p"},"nvidia")," and ",Object(c.b)("inlineCode",{parentName:"p"},"amd")," or a comma separated combination of both ",Object(c.b)("inlineCode",{parentName:"p"},"nvidia,amd"),".")))),Object(c.b)(i.a,{value:"eks",mdxType:"TabItem"},Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl init --provider eks \\\n  --enable-https \\\n  --enable-cert-manager \\\n  --dns-provider <dns-provider>\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"If you have GPU nodes, you need to set the ",Object(c.b)("inlineCode",{parentName:"p"},"--gpu-device-plugins")," flag. Valid values are ",Object(c.b)("inlineCode",{parentName:"p"},"nvidia")," and ",Object(c.b)("inlineCode",{parentName:"p"},"amd")," or a comma separated combination of both ",Object(c.b)("inlineCode",{parentName:"p"},"nvidia,amd"),".")))),Object(c.b)(i.a,{value:"gke",mdxType:"TabItem"},Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl init --provider gke \\\n  --enable-https \\\n  --enable-cert-manager \\\n  --dns-provider <dns-provider>\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"GKE automatically adds GPU device plugins to GPU nodes, so you do not have to set the ",Object(c.b)("inlineCode",{parentName:"p"},"--gpu-device-plugins")," flag."))))),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"The ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-https")," flag is optional and requires a TLS certificate, but it is highly recommended. You can optionally set the ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-cert-manager")," and ",Object(c.b)("inlineCode",{parentName:"p"},"--dns-provider")," flags, so TLS certificates are automatically created and renewed via ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://letsencrypt.org/"}),"Let's Encrypt"),". If you do not set this flag and your DNS provider isn't one of the ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"/docs/deployment/configuration/tls#supported-dns-providers"}),"supported DNS providers"),", then you have to create a wildcard certificate and manually manage it."))),Object(c.b)("ol",{start:3},Object(c.b)("li",{parentName:"ol"},Object(c.b)("p",{parentName:"li"},"Populate ",Object(c.b)("inlineCode",{parentName:"p"},"params.yaml")," by following the instructions in the template, you can also refer to the ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"/docs/deployment/configuration/files"}),"configuration files")," section.")),Object(c.b)("li",{parentName:"ol"},Object(c.b)("p",{parentName:"li"},"Finally, run the following command to deploy to your cluster:"))),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl apply\n")),Object(c.b)("ol",{start:5},Object(c.b)("li",{parentName:"ol"},"Once the deployment completes, the CLI will display the IP and wildcard domain you need to use to setup your DNS. You can also get this information again by running:")),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl app ip\n")),Object(c.b)("ol",{start:6},Object(c.b)("li",{parentName:"ol"},Object(c.b)("p",{parentName:"li"},"Create an ",Object(c.b)("inlineCode",{parentName:"p"},"A")," record in your DNS provider based on the instructions above.")),Object(c.b)("li",{parentName:"ol"},Object(c.b)("p",{parentName:"li"},"Use the following command to get your auth token to log into Onepanel:"))),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl auth token\n")))}d.isMDXComponent=!0},149:function(e,a,t){"use strict";t.d(a,"a",(function(){return d})),t.d(a,"b",(function(){return u}));var n=t(0),c=t.n(n);function o(e,a,t){return a in e?Object.defineProperty(e,a,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[a]=t,e}function i(e,a){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);a&&(n=n.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),t.push.apply(t,n)}return t}function r(e){for(var a=1;a<arguments.length;a++){var t=null!=arguments[a]?arguments[a]:{};a%2?i(Object(t),!0).forEach((function(a){o(e,a,t[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(t,a))}))}return e}function l(e,a){if(null==e)return{};var t,n,c=function(e,a){if(null==e)return{};var t,n,c={},o=Object.keys(e);for(n=0;n<o.length;n++)t=o[n],a.indexOf(t)>=0||(c[t]=e[t]);return c}(e,a);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)t=o[n],a.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(c[t]=e[t])}return c}var b=c.a.createContext({}),s=function(e){var a=c.a.useContext(b),t=a;return e&&(t="function"==typeof e?e(a):r(r({},a),e)),t},d=function(e){var a=s(e.components);return c.a.createElement(b.Provider,{value:a},e.children)},p={inlineCode:"code",wrapper:function(e){var a=e.children;return c.a.createElement(c.a.Fragment,{},a)}},m=c.a.forwardRef((function(e,a){var t=e.components,n=e.mdxType,o=e.originalType,i=e.parentName,b=l(e,["components","mdxType","originalType","parentName"]),d=s(t),m=n,u=d["".concat(i,".").concat(m)]||d[m]||p[m]||o;return t?c.a.createElement(u,r(r({ref:a},b),{},{components:t})):c.a.createElement(u,r({ref:a},b))}));function u(e,a){var t=arguments,n=a&&a.mdxType;if("string"==typeof e||n){var o=t.length,i=new Array(o);i[0]=m;var r={};for(var l in a)hasOwnProperty.call(a,l)&&(r[l]=a[l]);r.originalType=e,r.mdxType="string"==typeof e?e:n,i[1]=r;for(var b=2;b<o;b++)i[b]=t[b];return c.a.createElement.apply(null,i)}return c.a.createElement.apply(null,t)}m.displayName="MDXCreateElement"},150:function(e,a,t){var n;!function(){"use strict";var t={}.hasOwnProperty;function c(){for(var e=[],a=0;a<arguments.length;a++){var n=arguments[a];if(n){var o=typeof n;if("string"===o||"number"===o)e.push(n);else if(Array.isArray(n)&&n.length){var i=c.apply(null,n);i&&e.push(i)}else if("object"===o)for(var r in n)t.call(n,r)&&n[r]&&e.push(r)}}return e.join(" ")}e.exports?(c.default=c,e.exports=c):void 0===(n=function(){return c}.apply(a,[]))||(e.exports=n)}()},153:function(e,a,t){"use strict";var n=t(0);const c=Object(n.createContext)({tabGroupChoices:{},setTabGroupChoices:()=>{}});a.a=c},154:function(e,a,t){"use strict";var n=t(0),c=t.n(n);a.a=function(e){return c.a.createElement("div",null,e.children)}},156:function(e,a,t){"use strict";var n=t(0),c=t.n(n),o=t(153);var i=function(){return Object(n.useContext)(o.a)},r=t(150),l=t.n(r),b=t(93),s=t.n(b);const d=37,p=39;a.a=function(e){const{block:a,children:t,defaultValue:o,values:r,groupId:b}=e,{tabGroupChoices:m,setTabGroupChoices:u}=i(),[O,j]=Object(n.useState)(o);if(null!=b){const e=m[b];null!=e&&e!==O&&j(e)}const v=e=>{j(e),null!=b&&u(b,e)},g=[];return c.a.createElement("div",null,c.a.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:l()("tabs",{"tabs--block":a})},r.map(({value:e,label:a})=>c.a.createElement("li",{role:"tab",tabIndex:"0","aria-selected":O===e,className:l()("tabs__item",s.a.tabItem,{"tabs__item--active":O===e}),key:e,ref:e=>g.push(e),onKeyDown:e=>((e,a,t)=>{switch(t.keyCode){case p:((e,a)=>{const t=e.indexOf(a)+1;e[t]?e[t].focus():e[0].focus()})(e,a);break;case d:((e,a)=>{const t=e.indexOf(a)-1;e[t]?e[t].focus():e[e.length-1].focus()})(e,a)}})(g,e.target,e),onFocus:()=>v(e),onClick:()=>v(e)},a))),c.a.createElement("div",{role:"tabpanel",className:"margin-vert--md"},n.Children.toArray(t).filter(e=>e.props.value===O)[0]))}}}]);