data "google_compute_image" "debian" {
  family  = "debian-13"
  project = "debian-cloud"
}

resource "google_compute_disk" "disk" {
  for_each = var.hosts

  name  = "${each.key}-disk"
  image = data.google_compute_image.debian.self_link
  type  = each.value.disk_type
  size  = each.value.disk_size
}
