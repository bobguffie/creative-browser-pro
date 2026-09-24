# GitHub Setup — run these two commands yourself, then tell me "done"

## 1. Sign in to GitHub (one-time, ~30 seconds)

Open a terminal and run:

```bash
~/.local/bin/gh auth login --hostname github.com --git-protocol https --scopes repo --web
```

- It prints a **one-time code** (like `XXXX-XXXX`) and asks to press Enter.
- Press Enter → browser opens → paste the code → approve.
- Come back and tell me **"done"**.

## 2. That's it

Once you say done, I will automatically:

1. `git init` + commit the whole app (vendor/ AI engine included, node_modules excluded)
2. Create the **private** repo `creative-browser-pro` under your account
3. Push version **v2.0.0** with a proper release tag

*(If the browser doesn't open automatically, the command prints a URL — open it manually and enter the code.)*
