"""
Biomind — Etapa 3 (CLI): orientação de um caso pela linha de comando.
Usa a mesma lógica central do servidor web (biomind_core).

Uso:  python 03_answer.py "descricao do caso"
"""
import sys
import biomind_core as core


def main(pergunta):
    r = core.responder(pergunta)
    if r["status"] == "ok":
        print("\n" + r["answer"])
        print("\n" + "-" * 60 + "\nFontes consultadas:")
        for s in r["sources"]:
            print(f"  [{s['n']}] {s['source']} — pág. {s['pages']} (similaridade {s['sim']})")
        if r.get("disclaimer"):
            print("\n" + r["disclaimer"])
    else:
        print("\n" + r["message"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        main(" ".join(sys.argv[1:]))