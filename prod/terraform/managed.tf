locals {
  env = { for match in regexall("(?m)^(\\w+)=(.*)$", file("${path.module}/../cloud.env")) : match[0] => match[1] }
  postgres_host = one(flatten([
    for psc in google_sql_database_instance.postgres.settings[0].ip_configuration[0].psc_config :
    psc.psc_auto_connections[*].ip_address
  ]))
  redis_host = one([
    for connection in flatten(google_memorystore_instance.redis.endpoints[*].connections[*].psc_auto_connection[*]) :
    connection.ip_address if connection.connection_type == "CONNECTION_TYPE_PRIMARY"
  ])
}

resource "google_network_connectivity_service_connection_policy" "managed" {
  for_each = toset(["gcp-memorystore", "google-cloud-sql"])

  name          = "${each.key}-policy"
  location      = var.region
  service_class = each.key
  network       = google_compute_network.network.id

  psc_config {
    subnetworks = [google_compute_subnetwork.subnet.id]
  }
}

resource "google_sql_database_instance" "postgres" {
  database_version    = "POSTGRES_18"
  region              = var.region
  deletion_protection = false

  settings {
    edition           = "ENTERPRISE_PLUS"
    tier              = var.sql_tier
    availability_type = "ZONAL"

    location_preference {
      zone = var.zone
    }

    backup_configuration {
      enabled                        = false
      point_in_time_recovery_enabled = false
    }

    ip_configuration {
      ipv4_enabled = false
      ssl_mode     = "ALLOW_UNENCRYPTED_AND_ENCRYPTED"

      psc_config {
        psc_enabled               = true
        allowed_consumer_projects = [var.project]

        psc_auto_connections {
          consumer_network            = google_compute_network.network.id
          consumer_service_project_id = var.project
        }
      }
    }
  }

  depends_on = [google_network_connectivity_service_connection_policy.managed]
}

resource "google_sql_database" "database" {
  name            = local.env.POSTGRES_DB
  instance        = google_sql_database_instance.postgres.name
  deletion_policy = "ABANDON"
}

resource "google_sql_user" "user" {
  name            = local.env.POSTGRES_USER
  password        = local.env.POSTGRES_PASSWORD
  instance        = google_sql_database_instance.postgres.name
  deletion_policy = "ABANDON"
}

resource "google_memorystore_instance" "redis" {
  instance_id                 = "redis-auth"
  location                    = var.region
  mode                        = "CLUSTER_DISABLED"
  shard_count                 = 1
  replica_count               = 0
  node_type                   = var.redis_node_type
  engine_version              = "VALKEY_9_1"
  authorization_mode          = "AUTH_DISABLED"
  transit_encryption_mode     = "TRANSIT_ENCRYPTION_DISABLED"
  deletion_protection_enabled = false

  zone_distribution_config {
    mode = "SINGLE_ZONE"
    zone = var.zone
  }

  persistence_config {
    mode = "DISABLED"
  }

  desired_auto_created_endpoints {
    network    = google_compute_network.network.id
    project_id = var.project
  }

  depends_on = [google_network_connectivity_service_connection_policy.managed]
}
