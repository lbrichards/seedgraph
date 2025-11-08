"""12-domain taxonomy for seed topic coverage."""

DOMAINS = [
    "Science & Math",
    "Engineering & Tech",
    "Medicine & Biology",
    "Social Science & Economics",
    "Law & Policy",
    "History & Geography",
    "Arts & Literature",
    "Business & Finance",
    "Education & Study Skills",
    "Daily Life & Hobbies",
    "Sports & Games",
    "Environment & Energy"
]

# Tier-1 canonical topics (40 per domain)
TIER1_SEEDS = {
    "Science & Math": [
        "thermodynamics", "partial differential equations", "evolutionary biology",
        "quantum mechanics", "statistical mechanics", "topology", "group theory",
        "number theory", "complex analysis", "linear algebra", "calculus of variations",
        "differential geometry", "algebraic geometry", "combinatorics", "graph theory",
        "probability theory", "stochastic processes", "information theory", "chaos theory",
        "fractal geometry", "computational geometry", "numerical analysis", "optimization theory",
        "game theory", "control theory", "dynamical systems", "fluid dynamics",
        "electromagnetism", "optics", "nuclear physics", "particle physics",
        "astrophysics", "cosmology", "general relativity", "string theory",
        "molecular biology", "biochemistry", "genetics", "ecology", "neuroscience"
    ],

    "Engineering & Tech": [
        "distributed systems", "compilers", "FPGA design", "robotics kinematics",
        "computer architecture", "operating systems", "database systems", "networking protocols",
        "computer graphics", "signal processing", "image processing", "machine learning",
        "deep learning", "natural language processing", "computer vision", "reinforcement learning",
        "embedded systems", "real-time systems", "parallel computing", "cloud computing",
        "virtualization", "containerization", "microservices", "API design",
        "software testing", "version control", "CI/CD pipelines", "DevOps practices",
        "cybersecurity", "cryptography", "blockchain", "quantum computing",
        "analog circuits", "digital circuits", "VLSI design", "semiconductor physics",
        "control systems", "power electronics", "RF engineering", "antenna design"
    ],

    "Medicine & Biology": [
        "cardiovascular physiology", "microbiome", "immunology", "oncology",
        "neurology", "endocrinology", "pharmacology", "toxicology",
        "pathology", "radiology", "surgery", "anesthesiology",
        "psychiatry", "dermatology", "ophthalmology", "orthopedics",
        "pediatrics", "geriatrics", "obstetrics", "gynecology",
        "cell biology", "molecular genetics", "epigenetics", "proteomics",
        "genomics", "metabolomics", "bioinformatics", "systems biology",
        "developmental biology", "stem cell research", "regenerative medicine", "tissue engineering",
        "virology", "bacteriology", "parasitology", "mycology",
        "anatomy", "histology", "embryology", "physiology"
    ],

    "Social Science & Economics": [
        "microeconomics", "macroeconomics", "econometrics", "behavioral economics",
        "game theory applications", "industrial organization", "labor economics", "public economics",
        "development economics", "international trade", "monetary policy", "fiscal policy",
        "economic growth theory", "income inequality", "poverty analysis", "welfare economics",
        "cognitive psychology", "social psychology", "developmental psychology", "personality psychology",
        "psychometrics", "experimental psychology", "clinical psychology", "neuropsychology",
        "sociology of networks", "social stratification", "demography", "urban sociology",
        "political theory", "comparative politics", "international relations", "public policy",
        "anthropology", "linguistics", "semiotics", "cultural studies",
        "communication theory", "media studies", "journalism ethics", "propaganda analysis"
    ],

    "Law & Policy": [
        "constitutional law", "contract law", "tort law", "criminal law",
        "property law", "intellectual property", "patent law", "copyright law",
        "trademark law", "antitrust law", "securities law", "tax law",
        "environmental law", "labor law", "employment law", "immigration law",
        "family law", "administrative law", "civil procedure", "criminal procedure",
        "evidence law", "legal ethics", "jurisprudence", "legal theory",
        "international law", "human rights law", "humanitarian law", "trade law",
        "privacy law", "data protection", "cybersecurity law", "health law",
        "education policy", "housing policy", "transportation policy", "energy policy",
        "monetary policy", "fiscal policy", "regulatory policy", "social welfare policy"
    ],

    "History & Geography": [
        "ancient civilizations", "medieval history", "renaissance", "enlightenment",
        "industrial revolution", "colonialism", "imperialism", "world wars",
        "cold war", "decolonization", "globalization", "digital revolution",
        "economic history", "social history", "cultural history", "intellectual history",
        "military history", "diplomatic history", "political history", "religious history",
        "physical geography", "human geography", "economic geography", "political geography",
        "urban geography", "cultural geography", "historical geography", "regional geography",
        "cartography", "GIS", "remote sensing", "geomorphology",
        "climatology", "biogeography", "soil science", "hydrology",
        "archaeology", "paleontology", "historical linguistics", "numismatics"
    ],

    "Arts & Literature": [
        "classical literature", "modernist literature", "postmodern literature", "literary criticism",
        "poetry analysis", "narrative theory", "drama theory", "comparative literature",
        "creative writing", "screenwriting", "playwriting", "rhetoric",
        "classical music theory", "harmony", "counterpoint", "orchestration",
        "jazz theory", "electronic music", "music production", "sound design",
        "painting techniques", "drawing", "sculpture", "printmaking",
        "photography", "cinematography", "video production", "animation",
        "graphic design", "typography", "color theory", "composition",
        "architecture theory", "urban design", "landscape architecture", "interior design",
        "performance art", "installation art", "conceptual art", "art history"
    ],

    "Business & Finance": [
        "financial accounting", "managerial accounting", "cost accounting", "auditing",
        "corporate finance", "investment analysis", "portfolio theory", "asset pricing",
        "derivatives", "fixed income", "equity valuation", "risk management",
        "mergers and acquisitions", "corporate governance", "business ethics", "business law",
        "strategic management", "operations management", "supply chain management", "project management",
        "marketing strategy", "brand management", "consumer behavior", "market research",
        "organizational behavior", "human resource management", "leadership theory", "change management",
        "entrepreneurship", "venture capital", "private equity", "investment banking",
        "commercial banking", "insurance", "real estate finance", "international finance",
        "financial regulation", "monetary economics", "central banking", "cryptocurrency"
    ],

    "Education & Study Skills": [
        "pedagogical theory", "curriculum design", "assessment methods", "learning theories",
        "educational psychology", "cognitive development", "motivation theory", "metacognition",
        "instructional design", "e-learning", "blended learning", "flipped classroom",
        "active learning", "collaborative learning", "problem-based learning", "inquiry-based learning",
        "differentiated instruction", "universal design for learning", "inclusive education", "special education",
        "literacy education", "numeracy education", "STEM education", "language learning",
        "study strategies", "note-taking methods", "memory techniques", "time management",
        "critical thinking", "creative thinking", "analytical reasoning", "problem solving",
        "research methods", "academic writing", "citation practices", "information literacy",
        "test preparation", "exam strategies", "stress management", "learning disabilities"
    ],

    "Daily Life & Hobbies": [
        "nutrition science", "meal planning", "cooking techniques", "food safety",
        "fitness training", "strength training", "cardiovascular exercise", "flexibility training",
        "yoga practice", "meditation", "mindfulness", "stress reduction",
        "sleep hygiene", "circadian rhythms", "mental health", "emotional intelligence",
        "gardening", "horticulture", "composting", "permaculture",
        "woodworking", "metalworking", "3D printing", "electronics projects",
        "photography skills", "video editing", "graphic design", "digital art",
        "music practice", "instrument technique", "music theory basics", "songwriting",
        "language learning", "travel planning", "cultural etiquette", "navigation skills",
        "home maintenance", "DIY repairs", "organization systems", "budgeting"
    ],

    "Sports & Games": [
        "biomechanics", "sports physiology", "exercise science", "sports nutrition",
        "strength conditioning", "athletic training", "injury prevention", "sports psychology",
        "coaching methodology", "tactical analysis", "performance metrics", "video analysis",
        "football strategy", "basketball systems", "baseball analytics", "soccer tactics",
        "tennis technique", "golf biomechanics", "swimming technique", "track and field",
        "martial arts", "boxing technique", "wrestling", "judo",
        "chess strategy", "chess openings", "chess endgames", "chess tactics",
        "poker probability", "poker psychology", "game theory in poker", "tournament strategy",
        "esports strategy", "competitive gaming", "speedrunning", "game design",
        "board game mechanics", "card game design", "puzzle solving", "strategy optimization"
    ],

    "Environment & Energy": [
        "climate science", "atmospheric chemistry", "carbon cycle", "greenhouse effect",
        "climate modeling", "climate change impacts", "climate adaptation", "climate mitigation",
        "renewable energy", "solar energy", "wind energy", "hydroelectric power",
        "geothermal energy", "biomass energy", "ocean energy", "energy storage",
        "battery technology", "fuel cells", "hydrogen economy", "grid infrastructure",
        "energy efficiency", "building energy", "industrial energy", "transportation energy",
        "environmental chemistry", "water quality", "air quality", "soil contamination",
        "waste management", "recycling", "circular economy", "life cycle assessment",
        "conservation biology", "habitat restoration", "biodiversity", "ecosystem services",
        "sustainable agriculture", "agroecology", "organic farming", "precision agriculture",
        "environmental policy", "environmental law", "carbon pricing", "emissions trading"
    ]
}


def get_domain_target_share(total_seeds: int, num_domains: int = 12) -> int:
    """Calculate target number of seeds per domain."""
    return total_seeds // num_domains


def validate_tier1_coverage():
    """Validate that all domains have adequate Tier-1 seeds."""
    for domain, seeds in TIER1_SEEDS.items():
        assert len(seeds) >= 35, f"{domain} has only {len(seeds)} tier-1 seeds (need ≥35)"
    return True
