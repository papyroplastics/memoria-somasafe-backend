locals {
  registry_host = "${var.region}-docker.pkg.dev"
  registry      = "${local.registry_host}/${var.project}/${google_artifact_registry_repository.registry.repository_id}"
  files = {
    for file in [
      "compose.yaml",
      "compose.prod.yaml",
      "prod/cloud.env",
      "prod/postgres.conf",
      "prod/prometheus.cloud.yml",
    ] : "/opt/somasafe/backend/${file}" => file("${path.module}/../../${file}")
  }
  host_files = {
    for name, host in var.hosts : name => {
      "/usr/local/bin/somasafe-token"              = file("${path.module}/templates/somasafe-token.sh")
      "/etc/containers/containers.conf.d/dns.conf" = "[containers]\ndns_searches = [\"${var.zone}.c.${var.project}.internal\"]\n"
      "/usr/local/bin/somasafe-pull" = templatefile("${path.module}/templates/somasafe-pull.sh.tftpl", {
        images        = host.images
        registry      = local.registry
        registry_host = local.registry_host
        tag           = var.tag
        secret        = host.key ? google_secret_manager_secret.server_key.id : ""
      })
      "/usr/local/bin/somasafe-compose" = templatefile("${path.module}/templates/somasafe-compose.sh.tftpl", {
        profiles = host.profiles
        registry = local.registry
        tag      = var.tag
      })
      "/etc/systemd/system/somasafe.service" = templatefile("${path.module}/templates/somasafe.service.tftpl", {
        name = name
      })
    }
  }
}

resource "google_compute_instance" "instance" {
  for_each = var.hosts

  name                      = "${each.key}-inst"
  machine_type              = each.value.machine
  allow_stopping_for_update = true

  boot_disk {
    auto_delete = false
    device_name = "${each.key}-disk-bind"
    mode        = "READ_WRITE"
    source      = google_compute_disk.disk[each.key].self_link
  }

  network_interface {
    nic_type   = "GVNIC"
    stack_type = "IPV4_IPV6"
    subnetwork = google_compute_subnetwork.subnet.self_link

    ipv6_access_config {
      name         = "External IPv6"
      network_tier = "PREMIUM"
    }
  }

  service_account {
    email  = google_service_account.host.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    startup-script = templatefile("${path.module}/templates/startup.sh.tftpl", {
      files = merge(local.files, local.host_files[each.key])
    })
    enable-oslogin : "TRUE"
    enable-oslogin-2fa : "FALSE"
    VmDnsSetting : "ZonalOnly"
  }

  scheduling {
    provisioning_model          = "STANDARD"
    preemptible                 = false
    automatic_restart           = true
    on_host_maintenance         = "MIGRATE"
    instance_termination_action = "STOP"
    max_run_duration {
      seconds = var.max_run_hours * 3600
    }
  }

  reservation_affinity {
    type = "NO_RESERVATION"
  }

  depends_on = [google_compute_router_nat.nat]
}
