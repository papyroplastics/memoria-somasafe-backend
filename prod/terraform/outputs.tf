output "project" {
  value = var.project
}

output "registry" {
  value = local.registry
}

output "hosts" {
  value = join("\n", [for k, inst in google_compute_instance.instance :
  "${inst.name} ${inst.current_status} ${inst.network_interface[0].ipv6_access_config[0].external_ipv6}"])
}

output "internal_ipv4" {
  value = { for k, inst in google_compute_instance.instance : inst.name => inst.network_interface[0].network_ip }
}
