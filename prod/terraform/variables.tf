variable "project" {}

variable "region" {
  default = "us-central1"
}

variable "zone" {
  default = "us-central1-c"
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

variable "hosts" {
  type = map(object({
    machine   = string
    profiles  = list(string)
    images    = optional(list(string), [])
    disk_type = optional(string, "pd-standard")
    disk_size = optional(number, 10)
    key       = optional(bool, false)
  }))
  default = {
    fastapi-1  = { machine = "e2-medium", profiles = ["fastapi-1"], images = ["api"] }
    celery-1   = { machine = "e2-standard-4", profiles = ["celery-1"], images = ["worker"], disk_size = 30, key = true }
    postgres   = { machine = "e2-medium", profiles = ["postgres"], disk_type = "pd-ssd", disk_size = 60 }
    redis-auth = { machine = "e2-medium", profiles = ["redis-auth"] }
    client-1   = { machine = "e2-medium", profiles = ["client-1", "monitor"], images = ["bench"] }
  }
}
