#!/bin/bash

# Run the comprehensive site regeneration script
# This will:
# 1. Stop any running Jekyll processes
# 2. Clean the _site directory
# 3. Update the homepage news section
# 4. Synchronize news data with the news page
# 5. Rebuild the entire site
# 6. Start the Jekyll server
ruby scripts/regenerate_site.rb

# Note: The news.yml file watcher is now managed by the Jekyll plugin
# If you encounter synchronization issues, manually run:
# ruby scripts/update_home_news.rb 