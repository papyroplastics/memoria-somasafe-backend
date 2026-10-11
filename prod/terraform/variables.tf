variable "project" {}

variable "region" {
  default = "us-east1"
}

variable "zone" {
  default = "us-east1-c"
}

variable "subnet_cidr" {
  default = "10.0.0.0/24"
}

variable "max_run_hours" {
  default = 6
}

variable "tag" {
  default = "latest"
}

variable "sql_tier" {
  default = "db-perf-optimized-N-2"
}

variable "redis_node_type" {
  default = "HIGHMEM_MEDIUM"
}

variable "hosts" {
  type = map(object({
    machine   = string
    profiles  = list(string)
    images    = optional(list(string), [])
    disk_type = optional(string, "hyperdisk-balanced")
    disk_size = optional(number, 10)
    key       = optional(bool, false)
  }))
  default = {
    fastapi-1 = { machine = "c4-standard-4", profiles = ["fastapi-1"], images = ["api"] }
    celery-1  = { machine = "c4-standard-4", profiles = ["celery-1"], images = ["worker"], disk_size = 30, key = true }
    client-1  = { machine = "c4-standard-4", profiles = ["client-1", "monitor"], images = ["bench"] }
  }
}
