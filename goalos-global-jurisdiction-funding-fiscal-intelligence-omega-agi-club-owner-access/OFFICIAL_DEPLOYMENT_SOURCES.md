# Official deployment and hosting source spine

Verification date: 2026-07-24

## GitHub Pages

- GitHub Pages is a static-site hosting service for HTML, CSS and JavaScript: https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages
- To bypass Jekyll when publishing prebuilt static files from a branch, place an empty `.nojekyll` file at the root of the publishing source: https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site
- GitHub Pages limits and sensitive-transaction warning: https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits

## ENS

- Official Ethereum Mainnet ENS deployments, including Registry `0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e` and Name Wrapper `0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401`: https://docs.ens.domains/learn/deployments/
- ENS Registry ownership model: https://docs.ens.domains/registry/ens/
- Wrapped-name ownership model and `ownerOf()` routing: https://docs.ens.domains/resolvers/interacting/
- Name Wrapper overview: https://docs.ens.domains/wrapper/overview/
- Wrapped-name expiry: https://docs.ens.domains/wrapper/expiry/

## Release policy

The application hardcodes only the verified Ethereum Mainnet ENS Registry and official Mainnet Name Wrapper addresses listed above. Contract-deployment changes require a new dated release, source refresh and full QA rather than silent runtime mutation.
