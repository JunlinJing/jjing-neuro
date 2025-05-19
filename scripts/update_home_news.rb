#!/usr/bin/env ruby
require 'yaml'
require 'fileutils'

# Load news data
news_data_path = File.join(Dir.pwd, '_data', 'news.yml')

# Check if news.yml exists
unless File.exist?(news_data_path)
  puts "Error: News data file (_data/news.yml) not found!"
  exit 1
end

begin
  news_data = YAML.load_file(news_data_path)
rescue => e
  puts "Error loading YAML file: #{e.message}"
  exit 1
end

# Read home page template
template_path = File.join(Dir.pwd, '_templates', 'index_template.html')

# Create a template if it doesn't exist
unless File.exist?(template_path)
  # Ensure directory exists
  FileUtils.mkdir_p(File.dirname(template_path))

  # Copy current index.html as template
  index_path = File.join(Dir.pwd, 'index.html')
  if File.exist?(index_path)
    FileUtils.cp(index_path, template_path)
    puts "Created template file #{template_path}"
  else
    puts "Error: Home page file index.html not found"
    exit 1
  end
end

# Read template
begin
  template_content = File.read(template_path)
rescue => e
  puts "Error reading template file: #{e.message}"
  exit 1
end

# Process "other" updates
other_category = news_data.find { |cat| cat['category'] == 'other' }
other_items = other_category ? other_category['items'] : []
# Sort by timestamp field
other_items.sort_by! { |item| -item['timestamp'].to_i } if other_items.all? { |item| item['timestamp'] }

# Process "personal" updates
personal_category = news_data.find { |cat| cat['category'] == 'personal' }
personal_items = personal_category ? personal_category['items'] : []
# Sort by timestamp field
personal_items.sort_by! { |item| -item['timestamp'].to_i } if personal_items.all? { |item| item['timestamp'] }

# Generate other updates HTML
other_html = ''
other_items.take(3).each do |item|
  # Determine which URL to use (prefer external_url if available)
  url = item['external_url'] || item['url']
  
  # If using internal URL, add site.baseurl prefix
  if url && !url.start_with?('http') && !item['external_url']
    url = "{{ site.baseurl }}#{url}"
  end
  
  # Determine if link should open in new tab
  target = item['external_url'] ? ' target="_blank"' : ''
  
  # 不再为标题文本添加超链接
  content_html = item['content']
  
  other_html += <<-HTML
                    <div class="news-item">
                        <span class="date">#{item['date']}</span>
                        <p>#{content_html}</p>
                        <div class="news-tag">#{item['tag']}</div>
                        #{url ? "<a href=\"#{url}\" class=\"news-item-link\" aria-label=\"Read more about this news\"#{target}></a>" : ''}
                    </div>
  HTML
end

# Generate personal updates HTML
personal_html = ''
personal_items.take(3).each do |item|
  # Determine which URL to use (prefer external_url if available)
  url = item['external_url'] || item['url']
  
  # If using internal URL, add site.baseurl prefix
  if url && !url.start_with?('http') && !item['external_url']
    url = "{{ site.baseurl }}#{url}"
  end
  
  # Determine if link should open in new tab
  target = item['external_url'] ? ' target="_blank"' : ''
  
  # 不再为标题文本添加超链接
  content_html = item['content']
  
  personal_html += <<-HTML
                    <div class="news-item">
                        <span class="date">#{item['date']}</span>
                        <p>#{content_html}</p>
                        <div class="news-tag">#{item['tag']}</div>
                        #{url ? "<a href=\"#{url}\" class=\"news-item-link\" aria-label=\"Read more about this news\"#{target}></a>" : ''}
                    </div>
  HTML
end

# Check if the template contains the necessary markers
unless template_content.include?('<!-- BEGIN_OTHER_NEWS -->') && template_content.include?('<!-- END_OTHER_NEWS -->')
  puts "Error: Template does not contain required markers for 'Other News'"
  exit 1
end

unless template_content.include?('<!-- BEGIN_PERSONAL_NEWS -->') && template_content.include?('<!-- END_PERSONAL_NEWS -->')
  puts "Error: Template does not contain required markers for 'Personal News'"
  exit 1
end

# Replace other updates in the template
template_content = template_content.sub(/<!-- BEGIN_OTHER_NEWS -->.*?<!-- END_OTHER_NEWS -->/m, "<!-- BEGIN_OTHER_NEWS -->\n#{other_html}<!-- END_OTHER_NEWS -->")

# Replace personal updates in the template
template_content = template_content.sub(/<!-- BEGIN_PERSONAL_NEWS -->.*?<!-- END_PERSONAL_NEWS -->/m, "<!-- BEGIN_PERSONAL_NEWS -->\n#{personal_html}<!-- END_PERSONAL_NEWS -->")

# Save updated home page
index_path = File.join(Dir.pwd, 'index.html')
begin
  File.write(index_path, template_content)
  puts "Home page news section successfully updated!"
rescue => e
  puts "Error writing to index.html: #{e.message}"
  exit 1
end 