openenv-salary-negotiation/
├── app.py                  ← FastAPI REST server (step/reset/state/validate)
├── openenv.yaml            ← OpenEnv metadata spec
├── Dockerfile              ← HuggingFace Spaces ready
├── requirements.txt
├── README.md               ← Full docs with baseline scores
├── env/
│   ├── models.py           ← Pydantic: Action, Observation, Reward
│   └── negotiation_env.py  ← Core env with step/reset/state
├── tasks/
│   └── task_configs.py     ← Task 1/2/3 configs (easy→hard)
├── graders/
│   └── graders.py          ← Deterministic graders, 0.0–1.0
├── scripts/
│   └── baseline_agent.py   ← OpenAI API baseline runner
└── tests/
    └── test_env.py         ← 27 unit + integration tests
