# Jim Jing's Academic Website

This is my personal academic website built with Jekyll, focusing on neuroscience research and artificial intelligence work.

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

3. Visit `http://localhost:4000/jjing-neuro/`

## Deployment

The website is automatically deployed via GitHub Pages, accessible at: [https://junlinjing.github.io/jjing-neuro/](https://junlinjing.github.io/jjing-neuro/)

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
title: Jim Jing
bio: Neuroscience Researcher | AI Enthusiast | Exploring Brain and Cognition

# Social Media Links
twitter: JimJing1997
linkedin: jjunlin
github: JunlinJing

# Academic Links
researchgate: Junlin-Jing
orcid: 0009-0006-1290-7445
```

## License

This website is open source under the [MIT License](LICENSE).

## Acknowledgments

The website theme is modified from [Indigo](https://github.com/sergiokopplin/indigo). Thanks to the original author for their work.
