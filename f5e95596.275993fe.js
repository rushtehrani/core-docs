/*! For license information please see f5e95596.275993fe.js.LICENSE.txt */
(window.webpackJsonp=window.webpackJsonp||[]).push([[42],{143:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return l})),a.d(t,"metadata",(function(){return s})),a.d(t,"rightToc",(function(){return b})),a.d(t,"default",(function(){return d}));var n=a(2),o=a(6),c=(a(0),a(149)),r=a(156),i=a(154),l={title:"AKS deployment guide",sidebar_label:"AKS deployment"},s={id:"deployment/public/aks",title:"AKS deployment guide",description:"This document outlines the installation steps for Azure Kubernetes Service (AKS).",source:"@site/docs/deployment/public/aks.md",permalink:"/docs/deployment/public/aks",editUrl:"https://github.com/onepanelio/core-docs/tree/master/docs/deployment/public/aks.md",sidebar_label:"AKS deployment",sidebar:"deployment",previous:{title:"Overview",permalink:"/docs/deployment/overview"},next:{title:"EKS deployment guide",permalink:"/docs/deployment/public/eks"}},b=[{value:"Launch an AKS cluster",id:"launch-an-aks-cluster",children:[]},{value:"Install Onepanel",id:"install-onepanel",children:[]}],p={rightToc:b};function d(e){var t=e.components,a=Object(o.a)(e,["components"]);return Object(c.b)("wrapper",Object(n.a)({},p,a,{components:t,mdxType:"MDXLayout"}),Object(c.b)("p",null,"This document outlines the installation steps for Azure Kubernetes Service (AKS)."),Object(c.b)("h2",{id:"launch-an-aks-cluster"},"Launch an AKS cluster"),Object(c.b)("div",{className:"admonition admonition-important alert alert--info"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"important")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"Make sure ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest"}),"Azure CLI")," (",Object(c.b)("inlineCode",{parentName:"p"},"az"),") is installed before proceeding."))),Object(c.b)("p",null,"We recommend launching a cluster with 2 ",Object(c.b)("inlineCode",{parentName:"p"},"Standard_D4s_v3")," nodes to start, with autoscaling and network policy enabled. You can add additional CPU/GPU node pools as needed later."),Object(c.b)("p",null,"Here is a sample ",Object(c.b)("inlineCode",{parentName:"p"},"az")," command to create a bare minimum cluster:"),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"az aks create --resource-group <resource-group> --name <cluster-name> \\\n    --node-count 2 \\\n    --node-vm-size Standard_D4s_v3 \\\n    --node-osdisk-size 100 \\\n    --min-count 2 \\\n    --max-count 2 \\\n    --enable-cluster-autoscaler \\\n    --network-plugin azure \\\n    --network-policy azure \\\n    --enable-addons monitoring \\\n    --generate-ssh-keys\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"The ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-addons monitoring")," flag in the command above enables Azure Monitor for log aggregation which can incur additional charges. You can optionally remove this flag and add ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-efk-logging")," to ",Object(c.b)("inlineCode",{parentName:"p"},"opctl")," command below."))),Object(c.b)("p",null,"You can then get access credentials by running:"),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{}),"az aks get-credentials --resource-group <resource-group> --name <cluster-name> --admin\n")),Object(c.b)("h2",{id:"install-onepanel"},"Install Onepanel"),Object(c.b)("ol",null,Object(c.b)("li",{parentName:"ol"},"Download the latest ",Object(c.b)("inlineCode",{parentName:"li"},"opctl")," for your operating system from ",Object(c.b)("a",Object(n.a)({parentName:"li"},{href:"https://github.com/onepanelio/core/releases/latest"}),"our release page"),".")),Object(c.b)(r.a,{defaultValue:"linux",values:[{label:"Linux",value:"linux"},{label:"macOS",value:"macos"}],mdxType:"Tabs"},Object(c.b)(i.a,{value:"linux",mdxType:"TabItem"},Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"# Download the binary\ncurl -sLO https://github.com/onepanelio/core/releases/download/v0.9.0/opctl-linux-amd64\n\n# Make binary executable\nchmod +x opctl-linux-amd64\n\n# Move binary to path\nmv ./opctl-linux-amd64 /usr/local/bin/opctl\n\n# Test installation\nopctl version\n"))),Object(c.b)(i.a,{value:"macos",mdxType:"TabItem"},Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"# Download the binary\ncurl -sLO https://github.com/onepanelio/core/releases/download/v0.9.0/opctl-macos-amd64\n\n# Make binary executable\nchmod +x opctl-macos-amd64\n\n# Move binary to path\nmv ./opctl-macos-amd64 /usr/local/bin/opctl\n\n# Test installation\nopctl version\n")))),Object(c.b)("ol",{start:2},Object(c.b)("li",{parentName:"ol"},"Run the following command to initialize a ",Object(c.b)("inlineCode",{parentName:"li"},"params.yaml")," template for AKS:")),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl init --provider aks \\\n    --enable-https \\\n    --enable-cert-manager \\\n    --dns-provider <dns-provider>\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"The ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-https")," flag is optional and requires a TLS certificate, but it is highly recommended. You can optionally set the ",Object(c.b)("inlineCode",{parentName:"p"},"--enable-cert-manager")," and ",Object(c.b)("inlineCode",{parentName:"p"},"--dns-provider")," flags, so TLS certificates are automatically created and renewed via ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"https://letsencrypt.org/"}),"Let's Encrypt"),". If you do not set this flag and your DNS provider isn't one of the ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"/docs/deployment/configuration/tls#supported-dns-providers"}),"supported DNS providers"),", then you have to create a wildcard certificate and manually manage it."))),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"If you have GPU nodes, you need to set the ",Object(c.b)("inlineCode",{parentName:"p"},"--gpu-device-plugins")," flag. Valid values are ",Object(c.b)("inlineCode",{parentName:"p"},"nvidia")," and ",Object(c.b)("inlineCode",{parentName:"p"},"amd")," or a comma separated combination of both ",Object(c.b)("inlineCode",{parentName:"p"},"nvidia,amd"),"."))),Object(c.b)("ol",{start:3},Object(c.b)("li",{parentName:"ol"},Object(c.b)("p",{parentName:"li"},"Populate ",Object(c.b)("inlineCode",{parentName:"p"},"params.yaml")," by following the instructions in the template, you can also refer to ",Object(c.b)("a",Object(n.a)({parentName:"p"},{href:"/docs/deployment/configuration/files"}),"configuration files")," for more detailed information.")),Object(c.b)("li",{parentName:"ol"},Object(c.b)("p",{parentName:"li"},"Finally, run the following command to deploy to your cluster:"))),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl apply\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"If the command completes but it indicates that your cluster is not ready, you can check status again by running ",Object(c.b)("inlineCode",{parentName:"p"},"opctl app status"),". If you're still seeing issues, run ",Object(c.b)("inlineCode",{parentName:"p"},"kubectl get pods --all-namespaces")," to see if there are any crashing pods."))),Object(c.b)("ol",{start:5},Object(c.b)("li",{parentName:"ol"},"Once the deployment completes, the CLI will display the IP and wildcard domain you need to use to setup your DNS. You can also get this information again by running:")),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl app status\n")),Object(c.b)("ol",{start:6},Object(c.b)("li",{parentName:"ol"},"Create an ",Object(c.b)("inlineCode",{parentName:"li"},"A")," record in your DNS provider based on the instructions above.")),Object(c.b)("div",{className:"admonition admonition-tip alert alert--success"},Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(n.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(n.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"}),Object(c.b)("path",Object(n.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"})))),"tip")),Object(c.b)("div",Object(n.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"Note that you should use a wildcard ",Object(c.b)("inlineCode",{parentName:"p"},"A")," record, for example: ",Object(c.b)("inlineCode",{parentName:"p"},"*.example.com")," or ",Object(c.b)("inlineCode",{parentName:"p"},"*.subdomain.example.com")))),Object(c.b)("ol",{start:7},Object(c.b)("li",{parentName:"ol"},"Use the following command to get your auth token to log into Onepanel:")),Object(c.b)("pre",null,Object(c.b)("code",Object(n.a)({parentName:"pre"},{className:"language-bash"}),"opctl auth token\n")))}d.isMDXComponent=!0},149:function(e,t,a){"use strict";a.d(t,"a",(function(){return p})),a.d(t,"b",(function(){return u}));var n=a(0),o=a.n(n);function c(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function i(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){c(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function l(e,t){if(null==e)return{};var a,n,o=function(e,t){if(null==e)return{};var a,n,o={},c=Object.keys(e);for(n=0;n<c.length;n++)a=c[n],t.indexOf(a)>=0||(o[a]=e[a]);return o}(e,t);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);for(n=0;n<c.length;n++)a=c[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(o[a]=e[a])}return o}var s=o.a.createContext({}),b=function(e){var t=o.a.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):i(i({},t),e)),a},p=function(e){var t=b(e.components);return o.a.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return o.a.createElement(o.a.Fragment,{},t)}},m=o.a.forwardRef((function(e,t){var a=e.components,n=e.mdxType,c=e.originalType,r=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),p=b(a),m=n,u=p["".concat(r,".").concat(m)]||p[m]||d[m]||c;return a?o.a.createElement(u,i(i({ref:t},s),{},{components:a})):o.a.createElement(u,i({ref:t},s))}));function u(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var c=a.length,r=new Array(c);r[0]=m;var i={};for(var l in t)hasOwnProperty.call(t,l)&&(i[l]=t[l]);i.originalType=e,i.mdxType="string"==typeof e?e:n,r[1]=i;for(var s=2;s<c;s++)r[s]=a[s];return o.a.createElement.apply(null,r)}return o.a.createElement.apply(null,a)}m.displayName="MDXCreateElement"},150:function(e,t,a){var n;!function(){"use strict";var a={}.hasOwnProperty;function o(){for(var e=[],t=0;t<arguments.length;t++){var n=arguments[t];if(n){var c=typeof n;if("string"===c||"number"===c)e.push(n);else if(Array.isArray(n)&&n.length){var r=o.apply(null,n);r&&e.push(r)}else if("object"===c)for(var i in n)a.call(n,i)&&n[i]&&e.push(i)}}return e.join(" ")}e.exports?(o.default=o,e.exports=o):void 0===(n=function(){return o}.apply(t,[]))||(e.exports=n)}()},153:function(e,t,a){"use strict";var n=a(0);const o=Object(n.createContext)({tabGroupChoices:{},setTabGroupChoices:()=>{}});t.a=o},154:function(e,t,a){"use strict";var n=a(0),o=a.n(n);t.a=function(e){return o.a.createElement("div",null,e.children)}},156:function(e,t,a){"use strict";var n=a(0),o=a.n(n),c=a(153);var r=function(){return Object(n.useContext)(c.a)},i=a(150),l=a.n(i),s=a(93),b=a.n(s);const p=37,d=39;t.a=function(e){const{block:t,children:a,defaultValue:c,values:i,groupId:s}=e,{tabGroupChoices:m,setTabGroupChoices:u}=r(),[O,j]=Object(n.useState)(c);if(null!=s){const e=m[s];null!=e&&e!==O&&j(e)}const v=e=>{j(e),null!=s&&u(s,e)},h=[];return o.a.createElement("div",null,o.a.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:l()("tabs",{"tabs--block":t})},i.map(({value:e,label:t})=>o.a.createElement("li",{role:"tab",tabIndex:"0","aria-selected":O===e,className:l()("tabs__item",b.a.tabItem,{"tabs__item--active":O===e}),key:e,ref:e=>h.push(e),onKeyDown:e=>((e,t,a)=>{switch(a.keyCode){case d:((e,t)=>{const a=e.indexOf(t)+1;e[a]?e[a].focus():e[0].focus()})(e,t);break;case p:((e,t)=>{const a=e.indexOf(t)-1;e[a]?e[a].focus():e[e.length-1].focus()})(e,t)}})(h,e.target,e),onFocus:()=>v(e),onClick:()=>v(e)},t))),o.a.createElement("div",{role:"tabpanel",className:"margin-vert--md"},n.Children.toArray(a).filter(e=>e.props.value===O)[0]))}}}]);