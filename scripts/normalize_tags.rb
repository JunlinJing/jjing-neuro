#!/usr/bin/env ruby

require 'yaml'

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
  'signal-processing' => 'Signal Processing'
}

def normalize_tags(tags)
  return [] if tags.nil?
  
  tags.map do |tag|
    normalized = TAG_MAPPINGS[tag.downcase] || tag.capitalize
    normalized
  end
end

def process_file(file_path)
  content = File.read(file_path)
  
  # Split front matter and content
  if content =~ /\A(---\s*\n.*?\n?)^(---\s*$\n?)/m
    front_matter = YAML.load($1)
    content_remainder = content[$1.length + $2.length..-1]
    
    if front_matter['tags']
      front_matter['tags'] = normalize_tags(front_matter['tags'])
      
      # Write back to file
      File.write(file_path, "---\n#{front_matter.to_yaml}---#{content_remainder}")
      puts "Normalized tags in #{file_path}: #{front_matter['tags'].join(', ')}"
    end
  end
end

# Process all markdown files in _posts directory
Dir.glob('_posts/**/*.{md,markdown}') do |file|
  process_file(file)
end 