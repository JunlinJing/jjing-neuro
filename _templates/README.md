# Blog Post Templates

This directory contains templates for creating new blog posts for different categories.

## How to Use These Templates

1. Choose the appropriate template based on the type of post you want to create:
   - `essay-template.md` - For essays and opinion pieces
   - `tutorial-template.md` - For tutorials and how-to guides
   - `neuroscience-template.md` - For neuroscience research articles

2. Create a new file in the appropriate directory:
   - Essays: `_posts/essays/YYYY-MM-DD-title-slug.md`
   - Tutorials: `_posts/tutorials/YYYY-MM-DD-title-slug.md`
   - Neuroscience: `_posts/neuroscience/YYYY-MM-DD-title-slug.md`

3. Copy the content from the template to your new file

4. Update the front matter (the YAML section at the top between `---` marks):
   - Change the title
   - Set the correct date (format: YYYY-MM-DD)
   - Add appropriate tags
   - Write a brief description
   - Add an image path if applicable (and set headerImage to true)

5. Write your content, replacing the placeholder sections with your actual content

## Front Matter Fields

- `layout`: Always use "post" (don't change this)
- `title`: The title of your post
- `date`: Publication date in YYYY-MM-DD format
- `categories`: The category of your post (essays, tutorials, or neuroscience)
- `tags`: Keywords related to your post
- `author`: Your name or username
- `image`: Path to the header image (if applicable)
- `headerImage`: Set to true if you want to display the header image
- `description`: A brief summary of your post (1-2 sentences)

## Example Usage

To create a new tutorial about Python data visualization:

1. Create a new file: `_posts/tutorials/2024-06-01-python-data-visualization.md`
2. Copy the content from `tutorial-template.md`
3. Update the front matter:
   ```yaml
   ---
   layout: post
   title: "Data Visualization with Python: A Comprehensive Guide"
   date: 2024-06-01
   categories: 
     - tutorials
   tags:
     - python
     - data visualization
     - matplotlib
     - seaborn
   author: Jim Jing
   image: /assets/images/blog/2024/python-data-viz.jpg
   headerImage: true
   description: "A comprehensive guide to data visualization in Python using Matplotlib and Seaborn"
   ---
   ```
4. Write your tutorial content, replacing the placeholder sections

## Images

Place your images in the `/assets/images/blog/YYYY/` directory, where YYYY is the current year. 