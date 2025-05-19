#!/usr/bin/env ruby

require 'jekyll'
require 'listen'

task :default => [:serve]

task :update_news do
  puts "Updating home page news section..."
  system("ruby scripts/update_home_news.rb")
end

task :serve => :update_news do
  puts "Starting Jekyll server..."
  system("bundle exec jekyll serve")
end

task :build => :update_news do
  puts "Building site..."
  system("bundle exec jekyll build")
end

# Watch for changes in news.yml and update home page automatically
task :watch_news do
  puts "Starting to watch news.yml for changes..."
  news_file = File.join(Dir.pwd, '_data', 'news.yml')
  listener = Listen.to(File.dirname(news_file)) do |modified, added, removed|
    modified_files = modified.select { |file| file.end_with?('news.yml') }
    if !modified_files.empty?
      puts "Detected changes in news.yml, updating home page..."
      system("ruby scripts/update_home_news.rb")
    end
  end
  listener.start
  sleep
rescue Interrupt
  puts "Stopping watcher..."
end

# Run both Jekyll server and news watcher
task :auto => [:update_news, :watch_news]
