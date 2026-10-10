terraform {
  required_providers {
    google = {
      source  = "registry.terraform.io/hashicorp/google"
      version = "~>8.5.0"
    }
  }
}

provider "google" {
  project = var.project
  region  = var.region
  zone    = var.zone
}

