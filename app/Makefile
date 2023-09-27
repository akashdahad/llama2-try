# eval $(minikube docker-env)
APP_NAME=text-extractor
export DOCKER_BUILDKIT=1

start:
	docker-compose -p ${APP_NAME} -f docker-compose.yaml up -d --remove-orphans
stop:
	docker-compose -p ${APP_NAME} -f docker-compose.yaml down
build:
	docker-compose -p ${APP_NAME} -f docker-compose.yaml build
pull:
	docker-compose -p ${APP_NAME} -f docker-compose.yaml pull

login:
	az acr login --name bhyveregistry.azurecr.io

network-install:
	@if [ ! $$(docker network ls | awk '{print $$2}' | grep backend-net) ]; then \
		docker network create backend-net --driver bridge; \
		echo "Docker network  backend-net created"; \
	fi

network-clean:
	@if [ $$(docker network ls | awk '{print $$2}' | grep backend-net) ]; then \
		docker network rm backend-net; \
		echo "Docker network  backend-net removed"; \
	fi

start-proxy:
	docker run -d \
		--name traefik \
		--hostname traefik \
		--restart always \
		--oom-kill-disable=false \
		--domainname local \
		-p 80:80 \
		-p 443:443 \
		-p 8080:8080 \
		-v /var/run/docker.sock:/var/run/docker.sock:ro \
		--label "service.name=traefik" \
		--label "service.port=8080" \
		--label "traefik.enable=true" \
		--label "traefik.http.routers.api.rule=Host(\`traefik.local\`) && PathPrefix(\`/api\`) || PathPrefix(\`/dashboard\`)" \
		--label "traefik.http.routers.api.service=api@internal" \
		--label "traefik.http.routers.api.entrypoints=web-secure,web" \
		--label "traefik.http.services.api.loadbalancer.server.port=8080" \
		--label "traefik.docker.network=backend-net" \
		traefik:latest \
			--log.level=ERROR \
			--accesslog=true \
			--api=true \
			--api.dashboard=true \
			--api.insecure=true \
			--serverstransport.insecureskipverify=true \
			--global.checknewversion=false \
			--global.sendanonymoususage=false \
			--providers.docker=true \
			--providers.docker.endpoint=unix:///var/run/docker.sock \
			--providers.docker.watch=true \
			--providers.docker.exposedbydefault=false \
			--entrypoints.web=true \
			--entrypoints.web.address=:80 \
			--entrypoints.web-secure.address=:443

stop-proxy:
	docker stop traefik
	docker rm traefik