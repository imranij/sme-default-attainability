# Data

No data files are kept in this repository. The three public datasets are downloaded from their
original sources and placed under `data/raw/`. The scripts write their intermediate files to
`data/processed/`, and both folders are listed in `.gitignore`.

## Berka (PKDD'99 Discovery Challenge)

`code/fetch_data.sh` downloads the eight `.asc` files from the mirror at
<https://github.com/jlacko/berka-dataset>, which holds them unchanged from the challenge release,
and checks each one against its SHA-256 sum. The eight files land in `data/raw/berka/`, and the
original challenge page that the mirror copies is <https://sorry.vse.cz/~berka/challenge/pkdd1999/>.

## SBA National

The file is `SBAnational.csv` from Li, Mickel and Taylor (2018), "Should this loan be approved or
denied?", Journal of Statistics Education 26(1), <https://doi.org/10.1080/10691898.2018.1434342>.
It comes with the supplementary material of that article. Download it by hand and place it at

```
data/raw/sba/SBAnational.csv
```

The copy used for the results has 899,164 data rows and 27 columns, is 179,430,516 bytes long, and
has this SHA-256 sum:

```
0359128a0b7599e83e4c2e4dcdd781d9121a765237f98d9f0ec1ab3e7c522548  SBAnational.csv
```

## Home Credit Default Risk

The files come from the Kaggle competition "Home Credit Default Risk",
<https://www.kaggle.com/c/home-credit-default-risk>, which needs a Kaggle account and acceptance
of the competition rules. Five of its tables are used, and they go in `data/raw/home_credit/`
under their original names. The other tables of the competition are not read.

| file | data rows | SHA-256 |
|---|---|---|
| `application_train.csv` | 307,511 | `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f` |
| `bureau.csv` | 1,716,428 | `9d799143423f280720cf51c1bfbbab2a0422da8ff2763335bb30bf43155494f7` |
| `bureau_balance.csv` | 27,299,925 | `33e09f06174c26f0be6b8b7398886c69e7bf0abbb29b4122f7841ffe545729a9` |
| `installments_payments.csv` | 13,605,401 | `428c2e2496e4d6d697ee8270e98497e5213c41be16d882eed1bc95b133726797` |
| `POS_CASH_balance.csv` | 10,001,358 | `0e13bc573ffa8fc29b3f00d975e557143193a405d675b0e4694b06fbdffcb0cd` |

Once the files are in place, `sha256sum -c` against the sums above, or `Get-FileHash` in
PowerShell, confirms that they are the same copies.

## SaudiSME-Sim

The simulator is code, so it has no data file. `lib/src/data/saudisme_sim.py` generates the
synthetic firms from a fixed seed every time a script needs them. No Saudi institutional or
firm-level data enter it, and its sector and size shares are illustrative values that are not
taken from a published statistical release.
