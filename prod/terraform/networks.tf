resource "google_compute_network" "network" {
  name                         = "network"
  auto_create_subnetworks      = false
  enable_ula_internal_ipv6     = false
  routing_mode                 = "REGIONAL"
  bgp_best_path_selection_mode = "LEGACY"
}

resource "google_compute_firewall" "external_firewall" {
  name      = "external-ipv6-firewall"
  direction = "INGRESS"
  priority  = 1000
  network   = google_compute_network.network.self_link

  source_ranges = ["0::0/0"]

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_firewall" "internal_firewall" {
  for_each = {
    ipv4 = { range = google_compute_subnetwork.subnet.ip_cidr_range, icmp = "icmp" }
    ipv6 = { range = google_compute_subnetwork.subnet.external_ipv6_prefix, icmp = "58" }
  }

  name      = "internal-${each.key}-firewall"
  direction = "INGRESS"
  priority  = 1000
  network   = google_compute_network.network.self_link

  source_ranges = [each.value.range]

  allow {
    protocol = "tcp"
  }

  allow {
    protocol = "udp"
  }

  allow {
    protocol = each.value.icmp
  }
}

resource "google_compute_subnetwork" "subnet" {
  name                     = "subnet"
  network                  = google_compute_network.network.id
  stack_type               = "IPV4_IPV6"
  ip_cidr_range            = var.subnet_cidr
  ipv6_access_type         = "EXTERNAL"
  private_ip_google_access = true
}

resource "google_compute_router" "router" {
  name    = "router"
  network = google_compute_network.network.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "nat"
  router                             = google_compute_router.router.name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
