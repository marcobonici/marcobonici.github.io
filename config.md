<!--
Add here global page variables to use throughout your website.
-->
@def author = "Marco Bonici"
@def mintoclevel = 2

# Add here files or directories that should be ignored by Franklin, otherwise
# these files might be copied and, if markdown, processed by Franklin which
# you might not want. Indicate directories by ending the name with a `/`.
# Base files such as LICENSE.md and README.md are ignored by default.
@def ignore = ["node_modules/"]

# RSS (the website_{title, descr, url} must be defined to get RSS)
@def generate_rss = true
@def website_title = "Marco Bonici"
@def website_descr = "Personal website"
@def website_url   = "https://marcobonici.github.io/"

<!--
Add here global latex commands to use throughout your pages.
-->
\newcommand{\R}{\mathbb R}
\newcommand{\scal}[1]{\langle #1 \rangle}

\newcommand{\note}[1]{@@note @@title ⚠ Note@@ @@content #1 @@ @@}