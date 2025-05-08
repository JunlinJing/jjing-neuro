#!/usr/bin/env ruby

require 'yaml'
require 'date'

# Define standard tag mappings
TAG_MAPPINGS = {
  'python' => 'Python',
  'eeg' => 'EEG',
  'bci' => 'BCI',
  'tutorial' => 'Tutorial',
  'neuroscience' => 'Neuroscience',
  'psychology' => 'Psychology',
  'machine-learning' => 'Machine Learning',
  'deep-learning' => 'Deep Learning',
  'data-analysis' => 'Data Analysis',
  'signal-processing' => 'Signal Processing',
  'cognitive-science' => 'Cognitive Science',
  'brain-computer-interface' => 'Brain-Computer Interface',
  'artificial-intelligence' => 'Artificial Intelligence',
  'neural-networks' => 'Neural Networks'
}

def normalize_tags(tags)
  return [] if tags.nil?
  
  tags.map do |tag|
    normalized = TAG_MAPPINGS[tag.to_s.downcase.strip] || tag.to_s.strip.split(/[\s-]/).map(&:capitalize).join(' ')
    normalized
  end
end

def format_value(key, value)
  case key
  when 'date'
    value.to_s  # Keep date as string
  when 'tags', 'tag'
    "\n" + value.map { |v| "- #{v}" }.join("\n")
  when 'description', 'title'
    "\"#{value.to_s.gsub('"', '\\"')}\""
  else
    value.inspect
  end
end

def process_file(file_path)
  puts "Processing file: #{file_path}"
  content = File.read(file_path)
  
  # Split front matter and content
  if content =~ /\A(---\s*\n.*?\n?)^(---\s*$\n?)/m
    front_matter = YAML.safe_load($1)
    content_remainder = content[$1.length + $2.length..-1]
    
    tags = front_matter['tags'] || front_matter['tag']
    
    if front_matter && tags
      original_tags = tags.dup
      normalized_tags = normalize_tags(original_tags)
      
      if original_tags != normalized_tags
        # Write back to file only if tags were changed
        new_content = "---\n"
        front_matter.each do |key, value|
          if key == 'tags' || key == 'tag'
            new_content += "#{key}:#{format_value(key, normalized_tags)}\n"
          else
            new_content += "#{key}: #{format_value(key, value)}\n"
          end
        end
        new_content += "---#{content_remainder}"
        
        File.write(file_path, new_content)
        puts "Updated tags in #{file_path}:"
        puts "  Before: #{original_tags.join(', ')}"
        puts "  After:  #{normalized_tags.join(', ')}"
        puts "---"
      else
        puts "No changes needed for #{file_path}"
      end
    else
      puts "No tags found in #{file_path}"
    end
  else
    puts "No front matter found in #{file_path}"
  end
rescue => e
  puts "Error processing #{file_path}: #{e.message}"
  puts e.backtrace
end

# Process all markdown files in blog/_posts directory
Dir.glob('blog/_posts/**/*.{md,markdown}') do |file|
  process_file(file)
end

puts "\nTag normalization complete!" 