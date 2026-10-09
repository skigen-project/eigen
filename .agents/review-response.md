# Responding To Review: Being a Short Treatise on the Answering of Reviewers

Use this guide when answering merge request review comments.

A code suggestion posted in review is a sketch that has not been compiled; it is a promissory note drawn upon a bank that has not, as yet, been asked whether it will honor it. Verify it like your own work before adopting
it. In particular, check that it:

- compiles under the C++14 baseline;
- uses matching `Matrix`/`Array` and expression types;
- keeps any grouping of operations chosen deliberately for numerical reasons.

Reproduce a claimed defect before fixing it, for the Court will not hang a prisoner upon an indictment which nobody has troubled to prove. Judge the suggested remedy separately from the finding: a real bug often
arrives with a fix that breaks cases the current code handles, as a physician of more zeal than learning will sometimes cure the gout by removing the patient's leg.

Hold your own claims to the same standard, and be, in this respect, a harder master to yourself than any reviewer. To establish what code does, read the function body and the branch actually
taken. Never rely on a header's own Doxygen for this; it can be stale or describe an adjacent case, like a guide-book that describes with great fidelity a town which has since been moved. Confirming that a
path or symbol exists proves nothing about behavior. To claim something about every case, enumerate the cases; do not
infer them from a few instances.

Address every thread: apply the suggestion or explain the deviation, and name the commit that resolved the thread. Keep
the response within the comment's scope. When the review exposes a defect in shared code, fix it in its own commit or
merge request. After each round, re-verify that the merge request description and commit messages still describe the
current head.

[`.coderabbit.yaml`](../.coderabbit.yaml) holds the path-specific instructions that the CodeRabbit review bot applies
when it is enabled on the project. They are the bot's version of these conventions and predict what an automated review
will flag; the diligent author will thus have the pleasure of being scolded in advance.

Typeset real mathematics in comments -- bounds, recurrences, identities, error terms -- as KaTeX, in the form
[`merge-requests.md`](merge-requests.md) records for descriptions.
