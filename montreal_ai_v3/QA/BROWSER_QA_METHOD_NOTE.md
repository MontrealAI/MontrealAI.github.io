# Browser QA method — v11.0.0

Chromium rendering used `page.set_content()` with local files intercepted through the synthetic origin `https://montreal.local/`. Because the browser keeps a `set_content()` document on an `about:blank` origin, the production Content-Security-Policy meta element was removed **only from the in-memory QA copy** so its own styles, scripts and images could load. The production HTML files were not modified by that step, and their CSP is included in static review.

The execution environment blocks direct navigation to `file://`, localhost and the synthetic host. Therefore reciprocal EN/FR controls were verified by DOM-equivalent URL resolution, and every resolved counterpart was rendered independently in Chromium.

Results: 136 desktop renders passed, 136 mobile renders passed, 124/124 mobile menus passed, 8/8 representative interactions passed, and all 136 page directions across 68 bilingual pairs resolved to the exact counterpart.
