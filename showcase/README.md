# Showcase: static presentation page

A single static page that presents this repository's image-segmentation study (estimating a tree's age from the
annual rings in a photograph of a wood cross-section) for the portfolio. No build step, relative paths only.

## What is on the page

- The question: one ring per year, and why counting rings by hand is slow.
- The pipeline from `k-means.ipynb` and `Edge-Detection-filters_Abdelrahman18999/Edge Detection.ipynb`, with their
  parameters: grayscale, K-Means on colour (k = 6, 100 rounds or ε 0.85, 10 random restarts), Sobel, Scharr,
  Laplacian and Canny.
- A live comparison: pick k from 2 to 8 and drag across the slice to compare the photograph with the K-Means result.
- The edge-filter results side by side, what the study found, the five papers in `reference/` and the code it builds on.

## Images

The wood photographs in `Data/images` and the edge-detector folders are stock photographs with watermarks
(Photocase, Zoonar, Shutterstock), so the page does not use them. Instead the repository's own methods were run,
with the notebooks' parameters, on openly licensed photographs:

| File | Source | Licence |
| --- | --- | --- |
| `img/stage-*.webp`, `img/kmeans-*.webp`, `img/edge-*.webp` | "Populus rings Botanical Museum Helsinki", Antti Leppänen, Wikimedia Commons | CC BY 4.0 (processed versions under the same licence) |
| `img/slice2*.webp` | "Tree Rings", Paul VanDerWerf, Wikimedia Commons | CC BY 2.0 |
| `img/psm-1873.webp` | Popular Science Monthly vol. 3 (1873), from `Data/images` | Public domain |
| `img/sequoia.webp` | "The General Sherman tree - Sequoia National Park (33241877800)", m01229, Wikimedia Commons (cropped) | CC BY-SA 2.0 |
| `img/author.webp` | Author portrait | |

Credits are also printed in the page footer. The band at the top of the hero is one radius of the poplar slice,
pith to bark, with dark wood drawn in ink (grey values compared with a moving mean).

## Structure

```
site/
  index.html
  css/style.css
  js/main.js        reveals, the stage that follows the reading column, the K-Means comparison, the drag row
  img/              processed and found images (webp)
  fonts/            STIX Two Text (SIL Open Font License), self-hosted
  favicon.ico, favicon.svg, apple-touch-icon.png, icon-512.png
```

## Embedding in the portfolio

Copy `site/` into the portfolio, for example `portfolio/site/projects/wood-annual-rings/`. All paths are relative,
so it works from any sub-folder.

## Local preview

```bash
cd showcase
docker compose up -d      # container wood-ring-showcase, port 80 inside, 8092 on the host, network private-net
```
