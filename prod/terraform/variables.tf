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
  default = "db-perf-optimized-N-8"
}

variable "redis_node_type" {
  default = "HIGHMEM_MEDIUM"
}

variable "topology" {
  default = "1x1"
}

variable "topologies" {
  type = map(map(object({
    machine   = string
    profiles  = list(string)
    images    = optional(list(string), [])
    disk_type = optional(string, "hyperdisk-balanced")
    disk_size = optional(number, 10)
    key       = optional(bool, false)
  })))
  default = {
    "1x1" = {
      fastapi-1 = { machine = "c4-standard-4", profiles = ["fastapi-1"], images = ["api"] }
      celery-1  = { machine = "c4-standard-4", profiles = ["celery-1"], images = ["worker"], disk_size = 30, key = true }
      client-1  = { machine = "c4-standard-4", profiles = ["client-1", "monitor"], images = ["bench"] }
    }
    "2x1" = {
      fastapi-1 = { machine = "c4-standard-4", profiles = ["fastapi-1"], images = ["api"] }
      fastapi-2 = { machine = "c4-standard-4", profiles = ["fastapi-2"], images = ["api"] }
      celery-1  = { machine = "c4-standard-2", profiles = ["celery-1"], images = ["worker"], disk_size = 30, key = true }
      client-1  = { machine = "c4-standard-2", profiles = ["client-1", "monitor"], images = ["bench"] }
    }
    "4x1" = {
      fastapi-1 = { machine = "c4-standard-2", profiles = ["fastapi-1"], images = ["api"] }
      fastapi-2 = { machine = "c4-standard-2", profiles = ["fastapi-2"], images = ["api"] }
      fastapi-3 = { machine = "c4-standard-2", profiles = ["fastapi-3"], images = ["api"] }
      fastapi-4 = { machine = "c4-standard-2", profiles = ["fastapi-4"], images = ["api"] }
      celery-1  = { machine = "c4-standard-2", profiles = ["celery-1"], images = ["worker"], disk_size = 30, key = true }
      client-1  = { machine = "c4-standard-2", profiles = ["client-1", "monitor"], images = ["bench"] }
    }
  }
}
