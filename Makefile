#!make
.DEFAULT_GOAL := help
# include .env
# .EXPORT_ALL_VARIABLES:

LAN_ID=$(shell whoami | cut -d@ -f1 )
help:
	@echo $(MAKEFILE_LIST)
	@grep '^[a-zA-Z]' $(MAKEFILE_LIST) | \
	sort | \
	awk -F ':.*?## ' 'NF==2 {printf "\033[35m %-25s\033[0m %s\n", $$1, $$2}'
.PHONY: help

run_notebook:
	@echo "Starting notebook server"
	docker-compose -f docker-compose.dev.yml build notebook
	docker-compose -f docker-compose.dev.yml up notebook

run_app:
	@echo "Starting dash app"
	docker-compose -f docker-compose.dev.yml build app

build:
	docker-compose -f docker-compose.yml -p pix2pix-$(LAN_ID) build

rebuild:
	docker-compose -f docker-compose.yml -p pix2pix-$(LAN_ID) build --no-cache --pull


#Stand up all services
up: build
	docker-compose -f docker-compose.yml up -d
#Tear down all services
down:
	docker-compose -f docker-compose.yml down -v