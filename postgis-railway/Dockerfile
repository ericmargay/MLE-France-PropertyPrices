FROM postgis/postgis:16-3.4

# Set default environment variables (Railway will override these)
ENV POSTGRES_DB=${PGDATABASE:-railway}
ENV POSTGRES_USER=${PGUSER:-postgres}
ENV POSTGRES_PASSWORD=${PGPASSWORD:-postgres}

# Expose PostgreSQL port
EXPOSE 5432

# Use default PostGIS entrypoint
CMD ["postgres"]
