#!/usr/bin/env ruby
require 'yaml'
require 'fileutils'

# Step 1: Ensure all Jekyll processes are stopped
puts "Checking for running Jekyll processes..."
jekyll_pids = `ps aux | grep jekyll | grep -v grep | awk '{print $2}'`.split("\n")
unless jekyll_pids.empty?
  puts "Stopping Jekyll processes..."
  system("ps aux | grep jekyll | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null")
  sleep 2
end

# Step 2: Clean existing site output
puts "Removing existing _site directory..."
FileUtils.rm_rf(File.join(Dir.pwd, '_site'))

# Step 3: Update homepage news section
puts "Updating homepage news section..."
system("ruby #{File.join(Dir.pwd, 'scripts', 'update_home_news.rb')}")

# Step 4: Synchronize news.yml data with news.md page
puts "Synchronizing news data with news page..."
system("ruby #{File.join(Dir.pwd, 'scripts', 'sync_news_page.rb')}")

# Step 5: Rebuild the entire site in production mode (no auto-regeneration)
puts "Rebuilding entire site..."
system("bundle exec jekyll build")

# Step 6: Start the Jekyll server
puts "Starting Jekyll server..."
puts "Visit http://localhost:4000/jjing-neuro/ to view the site"
puts "Press Ctrl+C to stop the server when finished"
exec("bundle exec jekyll serve --port 4000") 