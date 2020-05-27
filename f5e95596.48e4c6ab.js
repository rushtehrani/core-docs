(window.webpackJsonp=window.webpackJsonp||[]).push([[35],{128:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return r})),n.d(t,"metadata",(function(){return c})),n.d(t,"rightToc",(function(){return i})),n.d(t,"default",(function(){return b}));var a=n(1),o=(n(0),n(135));const r={title:"AKS deployment guide",sidebar_label:"AKS deployment"},c={id:"deployment/public/aks",title:"AKS deployment guide",description:"This document outlines the installation steps for Azure Kubernetes Service (AKS).",source:"@site/docs/deployment/public/aks.md",permalink:"/docs/deployment/public/aks",editUrl:"https://github.com/onepanelio/core-docs/tree/master/docs/deployment/public/aks.md",sidebar_label:"AKS deployment",sidebar:"deployment",previous:{title:"Overview",permalink:"/docs/deployment/overview"},next:{title:"EKS deployment guide",permalink:"/docs/deployment/public/eks"}},i=[{value:"Launch an AKS cluster",id:"launch-an-aks-cluster",children:[]},{value:"Install Onepanel",id:"install-onepanel",children:[]}],l={rightToc:i};function b({components:e,...t}){return Object(o.b)("wrapper",Object(a.a)({},l,t,{components:e,mdxType:"MDXLayout"}),Object(o.b)("p",null,"This document outlines the installation steps for Azure Kubernetes Service (AKS)."),Object(o.b)("h2",{id:"launch-an-aks-cluster"},"Launch an AKS cluster"),Object(o.b)("p",null,"We recommend launching a cluster with 2 ",Object(o.b)("inlineCode",{parentName:"p"},"Standard_D4s_v3")," nodes to start, with autoscaling and network policy enabled. You can add additional CPU/GPU node pools as needed later."),Object(o.b)("p",null,"Here is a sample ",Object(o.b)("inlineCode",{parentName:"p"},"az")," command to create a bare minimum cluster:"),Object(o.b)("pre",null,Object(o.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"az aks create --resource-group <resource-group> --name <cluster-name> \\\n    --node-count 2 \\\n    --node-vm-size Standard_D4s_v3 \\\n    --node-osdisk-size 100 \\\n    --min-count 2 \\\n    --max-count 2 \\\n    --enable-cluster-autoscaler \\\n    --network-plugin azure \\\n    --network-policy azure \\\n    --enable-addons monitoring \\\n    --generate-ssh-keys\n")),Object(o.b)("div",{className:"admonition admonition-note"},Object(o.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(o.b)("h5",{parentName:"div"},Object(o.b)("div",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(o.b)("svg",Object(a.a)({parentName:"div"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(o.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(o.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(o.b)("p",{parentName:"div"},"The ",Object(o.b)("inlineCode",{parentName:"p"},"--enable-addons monitoring")," flag in the command above enables Azure Monitor for log aggregation which can incur additional charges. You can optionally remove this flag and add ",Object(o.b)("inlineCode",{parentName:"p"},"--enable-efk-logging")," to ",Object(o.b)("inlineCode",{parentName:"p"},"opctl")," command below."))),Object(o.b)("p",null,"You can then get access credentials by running:"),Object(o.b)("pre",null,Object(o.b)("code",Object(a.a)({parentName:"pre"},{}),"az aks get-credentials --resource-group <resource-group> --name <cluster-name> --admin\n")),Object(o.b)("h2",{id:"install-onepanel"},"Install Onepanel"),Object(o.b)("ol",null,Object(o.b)("li",{parentName:"ol"},Object(o.b)("p",{parentName:"li"},"Download the latest ",Object(o.b)("inlineCode",{parentName:"p"},"opctl")," for your operating system from ",Object(o.b)("a",Object(a.a)({parentName:"p"},{href:"https://github.com/onepanelio/core/releases/latest"}),"our release page"),".")),Object(o.b)("li",{parentName:"ol"},Object(o.b)("p",{parentName:"li"},"Run the following command to initialize a ",Object(o.b)("inlineCode",{parentName:"p"},"params.yaml")," template for AKS:"))),Object(o.b)("pre",null,Object(o.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"opctl init --provider aks \\\n    --enable-https \\\n    --enable-cert-manager \\\n    --dns-provider <dns-provider>\n")),Object(o.b)("div",{className:"admonition admonition-note"},Object(o.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(o.b)("h5",{parentName:"div"},Object(o.b)("div",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(o.b)("svg",Object(a.a)({parentName:"div"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(o.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(o.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(o.b)("p",{parentName:"div"},"The ",Object(o.b)("inlineCode",{parentName:"p"},"--enable-https")," flag is optional and requires a TLS certificate, but it is highly recommended. You can optionally set the ",Object(o.b)("inlineCode",{parentName:"p"},"--enable-cert-manager")," and ",Object(o.b)("inlineCode",{parentName:"p"},"--dns-provider")," flags, so TLS certificates are automatically created and renewed via ",Object(o.b)("a",Object(a.a)({parentName:"p"},{href:"https://letsencrypt.org/"}),"Let's Encrypt"),". If you do not set this flag and your DNS provider isn't one of the ",Object(o.b)("a",Object(a.a)({parentName:"p"},{href:"/docs/deployment/configuration/tls#supported-dns-providers"}),"supported DNS providers"),", then you have to create a wildcard certificate and manually manage it."))),Object(o.b)("div",{className:"admonition admonition-note"},Object(o.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(o.b)("h5",{parentName:"div"},Object(o.b)("div",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(o.b)("svg",Object(a.a)({parentName:"div"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(o.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(o.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(o.b)("p",{parentName:"div"},"If you have GPU nodes, you need to set the ",Object(o.b)("inlineCode",{parentName:"p"},"--gpu-device-plugins")," flag. Valid values are ",Object(o.b)("inlineCode",{parentName:"p"},"nvidia")," and ",Object(o.b)("inlineCode",{parentName:"p"},"amd")," or a comma separated combination of both ",Object(o.b)("inlineCode",{parentName:"p"},"nvidia,amd"),"."))),Object(o.b)("ol",{start:3},Object(o.b)("li",{parentName:"ol"},Object(o.b)("p",{parentName:"li"},"Populate ",Object(o.b)("inlineCode",{parentName:"p"},"params.yaml")," by following the instructions in the template, you can also refer to the ",Object(o.b)("a",Object(a.a)({parentName:"p"},{href:"/docs/deployment/configuration/files"}),"configuration files")," section.")),Object(o.b)("li",{parentName:"ol"},Object(o.b)("p",{parentName:"li"},"Finally, run the following command to deploy to your cluster:"))),Object(o.b)("pre",null,Object(o.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"opctl apply\n")),Object(o.b)("ol",{start:5},Object(o.b)("li",{parentName:"ol"},"Once the deployment completes, the CLI will display the IP and wildcard domain you need to use to setup your DNS. You can also get this information again by running:")),Object(o.b)("pre",null,Object(o.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"opctl app ip\n")),Object(o.b)("ol",{start:6},Object(o.b)("li",{parentName:"ol"},"Create an ",Object(o.b)("inlineCode",{parentName:"li"},"A")," record in your DNS provider based on the instructions above.")),Object(o.b)("div",{className:"admonition admonition-tip"},Object(o.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(o.b)("h5",{parentName:"div"},Object(o.b)("div",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(o.b)("svg",Object(a.a)({parentName:"div"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(o.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(o.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(o.b)("p",{parentName:"div"},"Note that you should use a wildcard ",Object(o.b)("inlineCode",{parentName:"p"},"A")," record, for example: ",Object(o.b)("inlineCode",{parentName:"p"},"*.example.com")," or ",Object(o.b)("inlineCode",{parentName:"p"},"*.subdomain.example.com")))),Object(o.b)("ol",{start:7},Object(o.b)("li",{parentName:"ol"},"Use the following command to get your auth token to log into Onepanel:")),Object(o.b)("pre",null,Object(o.b)("code",Object(a.a)({parentName:"pre"},{className:"language-bash"}),"opctl auth token\n")))}b.isMDXComponent=!0},135:function(e,t,n){"use strict";n.d(t,"a",(function(){return d})),n.d(t,"b",(function(){return u}));var a=n(0),o=n.n(a);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function c(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?c(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):c(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,o=function(e,t){if(null==e)return{};var n,a,o={},r=Object.keys(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)n=r[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var b=o.a.createContext({}),p=function(e){var t=o.a.useContext(b),n=t;return e&&(n="function"==typeof e?e(t):i({},t,{},e)),n},d=function(e){var t=p(e.components);return o.a.createElement(b.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return o.a.createElement(o.a.Fragment,{},t)}},m=Object(a.forwardRef)((function(e,t){var n=e.components,a=e.mdxType,r=e.originalType,c=e.parentName,b=l(e,["components","mdxType","originalType","parentName"]),d=p(n),m=a,u=d["".concat(c,".").concat(m)]||d[m]||s[m]||r;return n?o.a.createElement(u,i({ref:t},b,{components:n})):o.a.createElement(u,i({ref:t},b))}));function u(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var r=n.length,c=new Array(r);c[0]=m;var i={};for(var l in t)hasOwnProperty.call(t,l)&&(i[l]=t[l]);i.originalType=e,i.mdxType="string"==typeof e?e:a,c[1]=i;for(var b=2;b<r;b++)c[b]=n[b];return o.a.createElement.apply(null,c)}return o.a.createElement.apply(null,n)}m.displayName="MDXCreateElement"}}]);