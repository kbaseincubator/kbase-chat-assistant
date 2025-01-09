# kbase-chat-assistant
KBase Research Assistant is powered by multiple Language Model (LLM)-based Agents designed to interact conversationally with users. This Assistant will serve as a guide, facilitating users' navigation within the KBase platform. 

Uses Poetry to handle requirements 

## Installation

1. Setup a conda environment
2. Run poetry install 

## Deployment on SPIN at NERSC

TODO. For now, see the SpinUp docs and tutorials. You'll be making a pretty simple
deployment of a single node, with a single service, with a single ingress to get this
up and running.

## Updating a deployment on SPIN at NERSC
1. Build the new docker image
```
docker build --platform="linux/amd64" -t registry.nersc.gov/kbase/kbase-chat-assistant .
```
2. Push to the registry. You might need to login with your NERSC id, pw+2fa
```
docker push registry.nersc.gov/kbase/kbase-chat-assistant
```
3. Log in to https://rancher2.spin.nersc.gov
4. Click on the `Development` server
5. On the left side find "Workloads". Click that, then "Deployments" underneath.
6. Find the chat assistant (namespace: `knowledge-engine`, container: `chat-assistant`)
7. On the far right, hit the 3-dots menu, then `Redeploy`
8. It can take a while for the container to get the memo that it should shut down, so just let it go.

## TODOs

1. Add the Knowledge Graph
2. User interface

![KBaseChatAssistant](chat_asst_screenshot.pdf)
