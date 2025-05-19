#!/usr/bin/env ruby
require 'yaml'
require 'fileutils'

# Update the homepage news section on site build
Jekyll::Hooks.register :site, :after_reset do |site|
  # Force update on site reset
  update_home_news
end

# Ensure homepage is updated when news data changes
Jekyll::Hooks.register :site, :post_read do |site|
  news_path = File.join(site.source, '_data', 'news.yml')
  index_path = File.join(site.source, 'index.html')
  
  if File.exist?(news_path) && File.exist?(index_path)
    # Check if news.yml is newer than index.html
    if File.mtime(news_path) > File.mtime(index_path)
      puts "News data has been updated, updating home page..."
      update_home_news
    end
  end
end

def update_home_news
  puts "Automatically updating home page news section..."
  
  # Load news data
  news_data_path = File.join(Dir.pwd, '_data', 'news.yml')
  return unless File.exist?(news_data_path)
  
  news_data = YAML.load_file(news_data_path)

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
      return
    end
  end

  # Read template
  template_content = File.read(template_path)

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
    # Fix URL links, add site.baseurl prefix
    url = item['url']
    if url && !url.start_with?('http')
      url = "{{ site.baseurl }}#{url}"
    end
    
    other_html += <<-HTML
                    <div class="news-item">
                        <span class="date">#{item['date']}</span>
                        <p>#{item['content']}</p>
                        <div class="news-tag">#{item['tag']}</div>
                        #{url ? "<a href=\"#{url}\" class=\"news-item-link\" aria-label=\"Read more about this news\"></a>" : ''}
                    </div>
    HTML
  end

  # Generate personal updates HTML
  personal_html = ''
  personal_items.take(3).each do |item|
    # Fix URL links, add site.baseurl prefix
    url = item['url']
    if url && !url.start_with?('http')
      url = "{{ site.baseurl }}#{url}"
    end
    
    personal_html += <<-HTML
                    <div class="news-item">
                        <span class="date">#{item['date']}</span>
                        <p>#{item['content']}</p>
                        <div class="news-tag">#{item['tag']}</div>
                        #{url ? "<a href=\"#{url}\" class=\"news-item-link\" aria-label=\"Read more about this news\"></a>" : ''}
                    </div>
    HTML
  end

  # Replace other updates in the template
  template_content = template_content.sub(/<!-- BEGIN_OTHER_NEWS -->.*?<!-- END_OTHER_NEWS -->/m, "<!-- BEGIN_OTHER_NEWS -->\n#{other_html}<!-- END_OTHER_NEWS -->")

  # Replace personal updates in the template
  template_content = template_content.sub(/<!-- BEGIN_PERSONAL_NEWS -->.*?<!-- END_PERSONAL_NEWS -->/m, "<!-- BEGIN_PERSONAL_NEWS -->\n#{personal_html}<!-- END_PERSONAL_NEWS -->")

  # Save updated home page
  index_path = File.join(Dir.pwd, 'index.html')
  File.write(index_path, template_content)

  puts "Home page news section successfully updated!"
end 