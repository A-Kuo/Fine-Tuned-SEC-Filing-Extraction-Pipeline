"""Download or generate SEC filing dataset for fine-tuning.

For the prototype, generates realistic synthetic training data that mirrors
actual SEC filing structure. In production, replace with real EDGAR API pulls.

The key ML insight: QLoRA fine-tuning learns the *mapping* from unstructured
financial text → structured JSON. The training data format defines what the
model learns to extract, so getting this format right is critical.

Usage:
    python scripts/download_dataset.py --num_samples 100
    python scripts/download_dataset.py --num_samples 5000 --split both
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger
from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, get_project_root

console = Console()

# ─── Realistic company data for synthetic generation ─────────────────────────

COMPANIES = [
    {"name": "Apple Inc.", "ticker": "AAPL", "sector": "Technology",
     "revenue_range": (350e9, 400e9), "income_range": (90e9, 110e9)},
    {"name": "Microsoft Corporation", "ticker": "MSFT", "sector": "Technology",
     "revenue_range": (200e9, 250e9), "income_range": (70e9, 90e9)},
    {"name": "Alphabet Inc.", "ticker": "GOOGL", "sector": "Technology",
     "revenue_range": (280e9, 340e9), "income_range": (60e9, 80e9)},
    {"name": "Amazon.com, Inc.", "ticker": "AMZN", "sector": "Consumer Discretionary",
     "revenue_range": (500e9, 600e9), "income_range": (20e9, 40e9)},
    {"name": "NVIDIA Corporation", "ticker": "NVDA", "sector": "Technology",
     "revenue_range": (60e9, 130e9), "income_range": (25e9, 60e9)},
    {"name": "Tesla, Inc.", "ticker": "TSLA", "sector": "Automotive",
     "revenue_range": (80e9, 100e9), "income_range": (7e9, 15e9)},
    {"name": "Meta Platforms, Inc.", "ticker": "META", "sector": "Technology",
     "revenue_range": (110e9, 160e9), "income_range": (30e9, 50e9)},
    {"name": "JPMorgan Chase & Co.", "ticker": "JPM", "sector": "Financials",
     "revenue_range": (150e9, 180e9), "income_range": (40e9, 55e9)},
    {"name": "Johnson & Johnson", "ticker": "JNJ", "sector": "Healthcare",
     "revenue_range": (85e9, 100e9), "income_range": (15e9, 22e9)},
    {"name": "Procter & Gamble Company", "ticker": "PG", "sector": "Consumer Staples",
     "revenue_range": (78e9, 85e9), "income_range": (14e9, 16e9)},
    {"name": "Visa Inc.", "ticker": "V", "sector": "Financials",
     "revenue_range": (30e9, 36e9), "income_range": (15e9, 18e9)},
    {"name": "UnitedHealth Group Incorporated", "ticker": "UNH", "sector": "Healthcare",
     "revenue_range": (350e9, 400e9), "income_range": (20e9, 25e9)},
    {"name": "Exxon Mobil Corporation", "ticker": "XOM", "sector": "Energy",
     "revenue_range": (300e9, 400e9), "income_range": (30e9, 60e9)},
    {"name": "Berkshire Hathaway Inc.", "ticker": "BRK", "sector": "Financials",
     "revenue_range": (300e9, 370e9), "income_range": (60e9, 100e9)},
    {"name": "Walmart Inc.", "ticker": "WMT", "sector": "Consumer Staples",
     "revenue_range": (600e9, 670e9), "income_range": (12e9, 18e9)},
]

FILING_TYPES = ["10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"]
FILING_TYPE_WEIGHTS = [0.35, 0.40, 0.15, 0.05, 0.05]

# ─── Prompt templates ────────────────────────────────────────────────────────

FILING_TEMPLATES = [
    # Template 1: Standard 10-K style
    """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM {filing_type}

{filing_type_description}

For the fiscal year ended {fiscal_year_end}

Commission File Number: {commission_number}

{company_name}
(Exact name of registrant as specified in its charter)

{state_of_incorporation}
(State or other jurisdiction of incorporation)

{ein}
(I.R.S. Employer Identification No.)

{address}

ITEM 1. BUSINESS

{company_name} ("the Company") is a {sector} company. For the fiscal year ended {fiscal_year_end}, the Company reported total revenue of ${revenue} and net income of ${net_income}. Total assets were ${total_assets} and total liabilities were ${total_liabilities}. Earnings per share (diluted) were ${eps}.

{risk_factors_section}

{mda_section}""",

    # Template 2: More concise filing
    """SEC FILING - FORM {filing_type}

Registrant: {company_name} (Ticker: {ticker})
Filing Date: {filing_date}
Period of Report: {fiscal_year_end}
CIK: {cik}

FINANCIAL HIGHLIGHTS:

Revenue: ${revenue}
Net Income: ${net_income}
Total Assets: ${total_assets}
Total Liabilities: ${total_liabilities}
Diluted EPS: ${eps}
Shares Outstanding: {shares_outstanding}

{business_description}

{risk_factors_section}""",

    # Template 3: Press-release style
    """{company_name} ({ticker}) - {filing_type} Filing

Filed with the SEC on {filing_date}

{company_name} today filed its {filing_type} with the Securities and Exchange Commission for the period ending {fiscal_year_end}.

Key Financial Data:
- Revenue for the period: ${revenue}
- Net income: ${net_income}
- Total assets as of period end: ${total_assets}
- Total liabilities: ${total_liabilities}
- Diluted earnings per share: ${eps}

{business_description}

The filing is available at https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={filing_type}""",
]

RISK_FACTORS = [
    "The Company faces risks related to global supply chain disruptions, which could adversely affect our ability to deliver products on time.",
    "Changes in tax regulations across jurisdictions in which we operate could materially impact our financial results.",
    "Competition in the {sector} industry is intense, and failure to innovate could result in loss of market share.",
    "Cybersecurity threats pose a significant risk to our operations, customer data, and reputation.",
    "Foreign currency fluctuations may materially impact our reported financial results.",
    "Regulatory changes in key markets could require significant operational adjustments.",
]

MDA_SECTIONS = [
    "Management believes the Company is well-positioned for future growth driven by strong demand in core markets and continued investment in R&D.",
    "During the fiscal year, the Company experienced {revenue_growth} revenue growth year-over-year, driven primarily by {growth_driver}.",
    "Operating expenses increased by {expense_growth} primarily due to investments in technology infrastructure and talent acquisition.",
]

BUSINESS_DESCRIPTIONS = [
    "The Company operates globally across multiple segments, serving both consumer and enterprise customers.",
    "{company_name} is a leading provider of {sector_product} with operations spanning {num_countries} countries.",
    "Founded in {founding_year}, the Company has grown to become one of the largest {sector} firms globally.",
]


def _fmt_dollars(amount: float) -> str:
    """Format large dollar amounts like SEC filings do."""
    if amount >= 1e12:
        return f"{amount / 1e12:.1f} trillion"
    elif amount >= 1e9:
        return f"{amount / 1e9:.1f} billion"
    elif amount >= 1e6:
        return f"{amount / 1e6:.1f} million"
    else:
        return f"{amount:,.0f}"


def _generate_filing_id(company: dict, date: datetime) -> str:
    """Generate realistic SEC filing accession number."""
    filer_id = random.randint(100000, 999999)
    seq = random.randint(10000000, 99999999)
    return f"000{filer_id}-{date.strftime('%y')}-{seq}"


def generate_single_example(idx: int) -> dict:
    """Generate one training example: (filing_text, extracted_json) pair.

    This is the core data format for QLoRA fine-tuning. The model learns:
        input: raw SEC filing text → output: structured JSON extraction

    The prompt format uses instruction-tuning style:
        [INST] Extract structured data from this SEC filing: {text} [/INST]
        {json_output}
    """
    company = random.choice(COMPANIES)
    filing_type = random.choices(FILING_TYPES, weights=FILING_TYPE_WEIGHTS, k=1)[0]

    # Generate realistic financials
    revenue = random.uniform(*company["revenue_range"])
    net_income = random.uniform(*company["income_range"])
    total_assets = revenue * random.uniform(1.5, 3.0)
    total_liabilities = total_assets * random.uniform(0.4, 0.7)
    eps = net_income / random.uniform(1e9, 5e9)  # shares outstanding

    # Generate dates
    filing_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))
    fiscal_year_end = filing_date - timedelta(days=random.randint(30, 90))

    # Build filing text from template
    template = random.choice(FILING_TEMPLATES)
    risk_section = "RISK FACTORS:\n" + "\n".join(
        random.sample(RISK_FACTORS, k=random.randint(2, 4))
    ).format(sector=company["sector"])

    mda_section = "MANAGEMENT'S DISCUSSION AND ANALYSIS:\n" + random.choice(MDA_SECTIONS).format(
        revenue_growth=f"{random.uniform(2, 25):.1f}%",
        growth_driver=random.choice(["cloud services", "new product launches", "market expansion", "AI adoption"]),
        expense_growth=f"{random.uniform(3, 15):.1f}%",
    )

    business_desc = random.choice(BUSINESS_DESCRIPTIONS).format(
        company_name=company["name"],
        sector=company["sector"],
        sector_product=random.choice(["technology solutions", "financial services", "consumer products", "healthcare services"]),
        num_countries=random.randint(20, 180),
        founding_year=random.randint(1880, 2010),
    )

    filing_text = template.format(
        filing_type=filing_type,
        filing_type_description=f"ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934" if "K" in filing_type else "QUARTERLY REPORT",
        fiscal_year_end=fiscal_year_end.strftime("%B %d, %Y"),
        commission_number=f"001-{random.randint(10000, 99999)}",
        company_name=company["name"],
        ticker=company["ticker"],
        state_of_incorporation=random.choice(["Delaware", "California", "New York", "Nevada"]),
        ein=f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}",
        address=f"{random.randint(1, 9999)} {random.choice(['Innovation', 'Commerce', 'Technology', 'Corporate'])} {random.choice(['Drive', 'Boulevard', 'Way', 'Avenue'])}",
        cik=f"{random.randint(100000, 9999999):07d}",
        filing_date=filing_date.strftime("%Y-%m-%d"),
        revenue=_fmt_dollars(revenue),
        net_income=_fmt_dollars(net_income),
        total_assets=_fmt_dollars(total_assets),
        total_liabilities=_fmt_dollars(total_liabilities),
        eps=f"{eps:.2f}",
        shares_outstanding=f"{random.uniform(1, 10):.2f} billion",
        risk_factors_section=risk_section,
        mda_section=mda_section,
        business_description=business_desc,
        sector=company["sector"],
    )

    # Ground truth extraction (what the model should output)
    extraction = {
        "filing_id": _generate_filing_id(company, filing_date),
        "company_name": company["name"],
        "ticker": company["ticker"],
        "filing_type": filing_type,
        "date": filing_date.strftime("%Y-%m-%d"),
        "fiscal_year_end": fiscal_year_end.strftime("%Y-%m-%d"),
        "revenue": f"${_fmt_dollars(revenue)}",
        "net_income": f"${_fmt_dollars(net_income)}",
        "total_assets": f"${_fmt_dollars(total_assets)}",
        "total_liabilities": f"${_fmt_dollars(total_liabilities)}",
        "eps": f"${eps:.2f}",
        "sector": company["sector"],
    }

    # Format as instruction-tuning pair
    instruction = (
        "Extract structured financial data from the following SEC filing. "
        "Return a JSON object with: filing_id, company_name, ticker, filing_type, "
        "date, fiscal_year_end, revenue, net_income, total_assets, total_liabilities, eps, sector."
    )

    return {
        "id": f"sec-{idx:06d}",
        "instruction": instruction,
        "input": filing_text.strip(),
        "output": json.dumps(extraction, indent=2),
        "metadata": {
            "company": company["name"],
            "filing_type": filing_type,
            "template_idx": FILING_TEMPLATES.index(template),
        },
    }


def generate_dataset(
    num_samples: int,
    output_path: Path,
    seed: int = 42,
) -> None:
    """Generate full training dataset as JSONL.

    Each line is one training example with instruction/input/output format
    compatible with HuggingFace's SFTTrainer for QLoRA fine-tuning.
    """
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_samples} synthetic SEC filing examples...")

    with open(output_path, "w") as f:
        for i in tqdm(range(num_samples), desc="Generating"):
            example = generate_single_example(i)
            f.write(json.dumps(example) + "\n")

    file_size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Dataset saved: {output_path} ({file_size_mb:.1f} MB, {num_samples} examples)")
    console.print(f"[bold green][OK][/bold green] {output_path} ({num_samples} examples, {file_size_mb:.1f} MB)")


def generate_sample_filing(output_path: Path) -> None:
    """Generate a single example filing for quick testing."""
    random.seed(0)
    example = generate_single_example(0)

    with open(output_path, "w") as f:
        f.write(example["input"])

    # Also write the expected extraction alongside
    expected_path = output_path.with_suffix(".expected.json")
    with open(expected_path, "w") as f:
        json.dump(json.loads(example["output"]), f, indent=2)

    console.print(f"[bold green][OK][/bold green] Sample filing: {output_path}")
    console.print(f"[bold green][OK][/bold green] Expected extraction: {expected_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SEC filing dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of training examples to generate (default: 100)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to generate",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data reserved for test set (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()

    config = load_config()
    data_dir = Path(args.output_dir) if args.output_dir else get_project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    test_count = int(args.num_samples * args.test_ratio)
    train_count = args.num_samples - test_count

    if args.split in ("train", "both"):
        generate_dataset(train_count, data_dir / "sec_filings_train.jsonl", seed=args.seed)

    if args.split in ("test", "both"):
        generate_dataset(test_count, data_dir / "sec_filings_test.jsonl", seed=args.seed + 1)

    # Always generate a sample filing for quick inference testing
    generate_sample_filing(data_dir / "sample_10k.txt")

    console.print(f"\n[bold]Dataset summary:[/bold]")
    console.print(f"  Train: {train_count} examples")
    console.print(f"  Test:  {test_count} examples")
    console.print(f"  Sample filing: {data_dir / 'sample_10k.txt'}")


if __name__ == "__main__":
    main()
