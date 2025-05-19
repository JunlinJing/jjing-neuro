#!/usr/bin/env ruby
require 'yaml'
require 'fileutils'

# Load news data from YAML file
puts "Loading news data from _data/news.yml..."
news_data_path = File.join(Dir.pwd, '_data', 'news.yml')
news_data = YAML.load_file(news_data_path)

# Read the news.md file
news_md_path = File.join(Dir.pwd, '_pages', 'news.md')
news_md_content = File.read(news_md_path)

# Extract the header part (YAML front matter + style)
header_match = news_md_content.match(/\A(---.*?---.*?<\/style>)/m)
header = header_match ? header_match[1] : ""

# Extract the controls part
controls_match = news_md_content.match(/(<div class="news-controls">.*?<\/div>\s*<\/div>)/m)
controls = controls_match ? controls_match[1] : ""

# Generate news cards from _data/news.yml
news_html = '<div class="news-list" id="newsList">'

# Process personal updates
personal_category = news_data.find { |cat| cat['category'] == 'personal' }
personal_items = personal_category ? personal_category['items'] : []
personal_items.sort_by! { |item| -item['timestamp'].to_i } if personal_items.all? { |item| item['timestamp'] }

# Process other updates
other_category = news_data.find { |cat| cat['category'] == 'other' }
other_items = other_category ? other_category['items'] : []
other_items.sort_by! { |item| -item['timestamp'].to_i } if other_items.all? { |item| item['timestamp'] }

# Combine all items and sort by timestamp
all_items = personal_items + other_items
all_items.sort_by! { |item| -item['timestamp'].to_i } if all_items.all? { |item| item['timestamp'] }

# Generate HTML for each news item
all_items.each do |item|
  date = item['date']
  content = item['content']
  tag = item['tag'].downcase
  category = tag.downcase # Simplify category by using tag directly
  timestamp = item['timestamp']
  
  # Determine location if it exists in the content
  location = ""
  if item['location']
    location = %{<span><b>Location:</b> #{item['location']}</span>}
  end
  
  # 移除标题超链接，直接使用内容文本
  title_html = item['content']
  
  # Build the card HTML
  news_html += %{
    <div class="news-card" data-category="#{category}" data-timestamp="#{timestamp}">
        <span class="news-tag #{category}">#{item['tag']}</span>
        <div class="news-info">
            <span><b>Date:</b> #{date}</span>
            #{location}
        </div>
        <div class="news-title">
            #{title_html}
        </div>
  }
  
  # Add additional details if there are any
  if item['details'] && !item['details'].empty?
    news_html += %{
        <ul class="news-points">
    }
    
    # Process details, handling special references
    item['details'].each do |detail|
      detail_html = detail
      
      # Handle PHD Basics reference in detail text
      if content == "PhD Basics for International Doctoral Researchers" && item['external_url']
        if detail.include?("PhD Basics for International Doctoral Researchers")
          # Replace text with linked version
          detail_html = detail.gsub(/PhD Basics for International Doctoral Researchers/) do |match|
            %{<a href="#{item['external_url']}" target="_blank">#{match}</a>}
          end
        end
      end
      
      # Handle GitHub reference in detail text
      if content == "Website Created" && item['external_url']
        if detail.include?("GitHub")
          # Replace GitHub text with linked version  
          detail_html = detail.gsub(/GitHub/) do |match|
            %{<a href="#{item['external_url']}" target="_blank">#{match}</a>}
          end
        end
      end
      
      news_html += %{
            <li>#{detail_html}</li>
      }
    end
    
    news_html += %{
        </ul>
    }
  end
  
  # Add social sharing links
  news_html += %{
        <div class="news-social-share">
            <a href="#" class="share-twitter" title="Share on Twitter" target="_blank"><i class="fab fa-twitter"></i></a>
            <a href="#" class="share-linkedin" title="Share on LinkedIn" target="_blank"><i class="fab fa-linkedin"></i></a>
            <a href="#" class="share-facebook" title="Share on Facebook" target="_blank"><i class="fab fa-facebook"></i></a>
            <a href="#" class="share-wechat" title="Share on WeChat"><i class="fab fa-weixin"></i></a>
            <a href="#" class="share-xiaohongshu" title="Share on RED" target="_blank">
                <svg viewBox="0 0 40 40" width="1em" height="1em" fill="currentColor">
                    <rect x="0" y="0" width="40" height="40" rx="8" fill="currentColor"/>
                    <text x="50%" y="56%" text-anchor="middle" fill="#fff" font-size="16" font-family="Arial" dy=".3em" font-weight="bold" letter-spacing="1">RED</text>
                </svg>
            </a>
        </div>
    </div>
  }
end

news_html += "</div>"

# Add JavaScript for filtering, sorting, and search
js_script = %{
<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize
        updateNewsCount();
    });

    function filterNews() {
        const category = document.getElementById('newsFilter').value;
        const cards = document.querySelectorAll('.news-card');
        
        cards.forEach(card => {
            if (category === 'all' || card.dataset.category === category) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
        
        updateNewsCount();
    }

    function sortNews() {
        const sortOrder = document.getElementById('newsSort').value;
        const newsContainer = document.getElementById('newsList');
        const cards = Array.from(document.querySelectorAll('.news-card'));
        
        cards.sort((a, b) => {
            const aTime = parseInt(a.dataset.timestamp || '0');
            const bTime = parseInt(b.dataset.timestamp || '0');
            
            if (sortOrder === 'date-desc') {
                return bTime - aTime; // Newest first
            } else {
                return aTime - bTime; // Oldest first
            }
        });
        
        // Re-append cards in the sorted order
        cards.forEach(card => newsContainer.appendChild(card));
    }

    function searchNews() {
        const searchText = document.getElementById('newsSearch').value.toLowerCase();
        const cards = document.querySelectorAll('.news-card');
        
        cards.forEach(card => {
            const cardText = card.textContent.toLowerCase();
            if (cardText.includes(searchText)) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
        
        updateNewsCount();
    }

    function updateNewsCount() {
        const visibleCards = document.querySelectorAll('.news-card[style="display: block;"], .news-card:not([style*="display"])').length;
        const countElem = document.getElementById('newsCount');
        if (countElem) {
            countElem.textContent = visibleCards;
        }
    }
</script>
}

# Combine all parts
new_content = header + "\n\n" + controls + "\n\n" + news_html + "\n\n" + js_script

# Backup the original file
FileUtils.cp(news_md_path, "#{news_md_path}.bak")

# Write the new content
File.write(news_md_path, new_content)

puts "News page updated successfully!" 