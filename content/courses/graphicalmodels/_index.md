---
date: "2018-09-09T00:00:00Z"
draft: false
lastmod: "2018-09-09T00:00:00Z"
linktitle: Graphical Models
menu:
  graphicalmodels:
    name: Overview
    weight: 1
summary: Introduction to graphical models (*in construction*)
title: Overview
toc: true
type: docs
weight: 1
---

## Test a local link
<!---
{{< icon name="download" pack="fas" >}}  my {{< staticref "media/resume-cambroise.pdf" "newtab" >}}resumé{{< /staticref >}}.
--->


[Queens University](https://www.queensu.ca)

<a   href="example1.md"> bibi </a>

## Flexibility 
http://127.0.0.1:4321/courses/graphicalmodels/example1.md
http://127.0.0.1:4321/courses/graphicalmodels/example1/
http://127.0.0.1:4321/courses/graphicalmodels/example1.md
This feature can be used for publishing content such as:

* **Online courses**
* **Project or software documentation**
* **Tutorials**

The `courses` folder may be renamed. For example, we can rename it to `docs` for software/project documentation or `tutorials` for creating an online course.

## Delete tutorials

**To remove these pages, delete the `courses` folder and see below to delete the associated menu link.**

## Update site menu

After renaming or deleting the `courses` folder, you may wish to update any `[[main]]` menu links to it by editing your menu configuration at `config/_default/menus.toml`.

For example, if you delete this folder, you can remove the following from your menu configuration:

```toml
[[main]]
  name = "Courses"
  url = "courses/"
  weight = 50
```

Or, if you are creating a software documentation site, you can rename the `courses` folder to `docs` and update the associated *Courses* menu configuration to:

```toml
[[main]]
  name = "Docs"
  url = "docs/"
  weight = 50
```

## Update the docs menu

If you use the *docs* layout, note that the name of the menu in the front matter should be in the form `[menu.X]` where `X` is the folder name. Hence, if you rename the `courses/example/` folder, you should also rename the menu definitions in the front matter of files within `courses/example/` from `[menu.example]` to `[menu.<NewFolderName>]`.


