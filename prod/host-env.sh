while IFS= read -r line; do
  case $line in
    [A-Za-z_]*=*) [ -n "$(printenv "${line%%=*}")" ] || export "$line" ;;
  esac
done < "$dir/cloud.env"
export POSTGRES_HOST=localhost REDIS_HOST=localhost
