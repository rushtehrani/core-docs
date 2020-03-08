(window.webpackJsonp=window.webpackJsonp||[]).push([[21],{112:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return i})),n.d(t,"metadata",(function(){return c})),n.d(t,"rightToc",(function(){return l})),n.d(t,"default",(function(){return p}));var a=n(1),o=n(6),r=(n(0),n(119)),i={title:"GKE installation guide",sidebar_label:"Installing on GKE"},c={id:"installation-guides/gke",title:"GKE installation guide",description:"This document outlines the installation steps for Google Kubernetes Engine (GKE).",source:"@site/docs/installation-guides/gke.md",permalink:"/core-docs/docs/installation-guides/gke",editUrl:"https://github.com/onepanelio/core-docs/tree/master/docs/installation-guides/gke.md",sidebar_label:"Installing on GKE",sidebar:"someSidebar",previous:{title:"EKS installation guide",permalink:"/core-docs/docs/installation-guides/eks"},next:{title:"Linux installation guide",permalink:"/core-docs/docs/installation-guides/linux"}},l=[{value:"Launch a GKE cluster",id:"launch-a-gke-cluster",children:[]},{value:"Install Onepanel Core",id:"install-onepanel-core",children:[]}],s={rightToc:l};function p(e){var t=e.components,n=Object(o.a)(e,["components"]);return Object(r.b)("wrapper",Object(a.a)({},s,n,{components:t,mdxType:"MDXLayout"}),Object(r.b)("p",null,"This document outlines the installation steps for Google Kubernetes Engine (GKE)."),Object(r.b)("h2",{id:"launch-a-gke-cluster"},"Launch a GKE cluster"),Object(r.b)("p",null,"We recommend launching a cluster with 2 ",Object(r.b)("inlineCode",{parentName:"p"},"n1-standard-4")," nodes to start, with autoscaling enabled and network policy enabled. You can add additional CPU/GPU node pools as needed later."),Object(r.b)("p",null,"Example ",Object(r.b)("inlineCode",{parentName:"p"},"gcloud")," script:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),'gcloud beta container --project $PROJECTNAME \\\n    clusters create $CLUSTERNAME --zone "us-central1-a" \\\n    --no-enable-basic-auth --release-channel "regular" \\\n    --machine-type "n1-standard-4" --image-type "COS" \\\n    --disk-type "pd-standard" --disk-size "100" \\\n    --metadata disable-legacy-endpoints=true \\\n    --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \\\n    --num-nodes $NUMNODES --enable-stackdriver-kubernetes \\\n    --enable-ip-alias --network $NETWORK --subnetwork $SUBNETWORK \\\n    --default-max-pods-per-node "110" --enable-network-policy \\\n    --addons HorizontalPodAutoscaling,HttpLoadBalancing --enable-autoupgrade \\\n    --enable-autorepair \n')),Object(r.b)("div",{className:"admonition admonition-important"},Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("div",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(a.a)({parentName:"div"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(r.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"important")),Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"Onepanel uses ",Object(r.b)("inlineCode",{parentName:"p"},"cert-manager")," to automatically renew TLS certificates. You need to setup additional firewall rules in GKE for this to work, refer to ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"https://cert-manager.io/docs/installation/compatibility/#gke"}),"cert-manager compatibility")," for more information."))),Object(r.b)("h2",{id:"install-onepanel-core"},"Install Onepanel Core"),Object(r.b)("p",null,"Download the latest ",Object(r.b)("inlineCode",{parentName:"p"},"opctl")," for your operating system from ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/onepanelio/cli/releases"}),"our release page")),Object(r.b)("p",null,"Run the following command to create ",Object(r.b)("inlineCode",{parentName:"p"},"params.yaml")," file for GKE:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"opctl init --provider gke --dns-provider <dns-provider> --logging\n")),Object(r.b)("div",{className:"admonition admonition-note"},Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("div",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(a.a)({parentName:"div"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(r.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"For a list of supported ",Object(r.b)("inlineCode",{parentName:"p"},"--dns-provider")," values see ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:""}),"CLI documentation")," "))),Object(r.b)("p",null,"Populate ",Object(r.b)("inlineCode",{parentName:"p"},"params.yaml")," as outlined in ",Object(r.b)("a",Object(a.a)({parentName:"p"},{href:"installation-guides/common"}),"common installation guide")),Object(r.b)("p",null,"Then run the following command to deploy to your cluster:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"opctl deploy\n")),Object(r.b)("p",null,"Once deployment completes, run the following command to get the external IP of Onepanel's gateway:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"kubectl get service istio-ingressgateway -n istio-system\n")),Object(r.b)("p",null,"This is the IP address you need to point your FQDN to in your DNS provider."),Object(r.b)("p",null,"Once deployment is complete, use the follownig command to get your auth token to log into Onepanel:"),Object(r.b)("pre",null,Object(r.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"opctl auth token\n")),Object(r.b)("div",{className:"admonition admonition-important"},Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(r.b)("h5",{parentName:"div"},Object(r.b)("div",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(r.b)("svg",Object(a.a)({parentName:"div"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(r.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"important")),Object(r.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(r.b)("p",{parentName:"div"},"For GKE, you have to run a simple ",Object(r.b)("inlineCode",{parentName:"p"},"kubectl")," (i.e. ",Object(r.b)("inlineCode",{parentName:"p"},"kubectl get nodes"),") for the above command to work."))))}p.isMDXComponent=!0},119:function(e,t,n){"use strict";n.d(t,"a",(function(){return b})),n.d(t,"b",(function(){return u}));var a=n(0),o=n.n(a);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function c(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,o=function(e,t){if(null==e)return{};var n,a,o={},r=Object.keys(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var s=o.a.createContext({}),p=function(e){var t=o.a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):c({},t,{},e)),n},b=function(e){var t=p(e.components);return o.a.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return o.a.createElement(o.a.Fragment,{},t)}},m=Object(a.forwardRef)((function(e,t){var n=e.components,a=e.mdxType,r=e.originalType,i=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),b=p(n),m=a,u=b["".concat(i,".").concat(m)]||b[m]||d[m]||r;return n?o.a.createElement(u,c({ref:t},s,{components:n})):o.a.createElement(u,c({ref:t},s))}));function u(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var r=n.length,i=new Array(r);i[0]=m;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:a,i[1]=c;for(var s=2;s<r;s++)i[s]=n[s];return o.a.createElement.apply(null,i)}return o.a.createElement.apply(null,n)}m.displayName="MDXCreateElement"}}]);