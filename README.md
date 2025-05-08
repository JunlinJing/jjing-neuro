# Academic Website

This is an academic website built with Jekyll, focusing on research and academic work.

## Website Content

- **About**: Personal introduction, educational background, research experience
- **Research**: Current research projects, methodologies, collaboration opportunities
- **Publications**: Journal articles, conference papers, preprints
- **Projects**: Open source projects, technical contributions
- **Blog**: Technical tutorials, research notes, academic thoughts

## Local Development

1. Install dependencies:
```bash
# Install Ruby and Bundler
gem install bundler
# Install project dependencies
bundle install
```

2. Run locally:
```bash
bundle exec jekyll serve
```

3. Visit `http://localhost:4000`

## Deployment

The website is automatically deployed via GitHub Pages.

## Tech Stack

- [Jekyll](https://jekyllrb.com/) - Static site generator
- [SASS](https://sass-lang.com/) - CSS preprocessor
- Responsive Design - Mobile and desktop compatible
- Dark/Light Theme - Supports dark/light theme switching

## File Structure

```
.
├── _includes/          # Reusable template components
├── _layouts/          # Page layout templates
├── _pages/           # Main page content
├── _posts/           # Blog posts
├── assets/           # Static assets (images, styles, etc.)
└── _config.yml       # Website configuration file
```

## Configuration

Main configuration in `_config.yml`:

```yaml
# Basic Information
title: Your Name
bio: Your Title | Your Field | Your Focus

# Social Media Links
twitter: username
linkedin: username
github: username

# Academic Links
researchgate: username
orcid: xxxx-xxxx-xxxx-xxxx
```

## License

This website is open source under the [MIT License](LICENSE).

## Acknowledgments

The website theme is modified from [Indigo](https://github.com/sergiokopplin/indigo). Thanks to the original author for their work.
