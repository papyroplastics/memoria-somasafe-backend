resource "google_artifact_registry_repository" "registry" {
  repository_id = "somasafe"
  location      = var.region
  format        = "DOCKER"
}

resource "google_secret_manager_secret" "server_key" {
  secret_id = "server-private-key"

  replication {
    auto {}
  }
}

resource "google_service_account" "host" {
  account_id   = "somasafe-host"
  display_name = "SomaSafe benchmark hosts"
}

resource "google_artifact_registry_repository_iam_member" "host_reader" {
  location   = google_artifact_registry_repository.registry.location
  repository = google_artifact_registry_repository.registry.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.host.email}"
}

resource "google_secret_manager_secret_iam_member" "host_key" {
  secret_id = google_secret_manager_secret.server_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.host.email}"
}
