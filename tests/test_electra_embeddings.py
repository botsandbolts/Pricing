import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pricing.electra_embeddings import (
    MODEL_ID, SHARDS, DIMENSIONS, MAX_TOKENS, POOLING,
    masked_mean, shard_positions, merge_shards,
)


class ElectraFeaturesTest(unittest.TestCase):
    def test_padding_is_excluded_and_vectors_are_normalized(self):
        hidden = np.array([[[1, 2], [3, 4], [1000, -1000]]], dtype=np.float32)
        actual = masked_mean(hidden, [[1, 1, 0]])
        expected = np.array([[2, 3]]) / np.sqrt(13)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        np.testing.assert_allclose(actual, masked_mean(hidden[:, :2], [[1, 1]]))

    def test_bad_pooling_inputs_fail(self):
        for hidden, mask in [
            (np.ones((1, 2, 3)), [[0, 0]]),
            (np.ones((1, 2, 3)), [[1]]),
            (np.full((1, 2, 3), np.nan), [[1, 1]]),
            (np.zeros((1, 2, 3)), [[1, 1]]),
        ]:
            with self.assertRaises(ValueError):
                masked_mean(hidden, mask)

    def test_twenty_five_shards_cover_each_row_once_across_reader_chunks(self):
        n = 2073
        assignment = np.zeros(n, dtype=int)
        for shard in range(SHARDS):
            positions = []
            for offset in range(0, n, 1000):
                size = min(1000, n - offset)
                indices = np.arange(offset, offset + size)
                positions.extend(indices[shard_positions(offset, size, shard)])
            expected = list(range(shard, n, SHARDS))
            self.assertEqual(positions, expected)
            assignment[positions] += 1
        np.testing.assert_array_equal(assignment, np.ones(n))
        with self.assertRaises(ValueError):
            shard_positions(0, 3, SHARDS)


class ElectraShardMergeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = {
            "model": MODEL_ID, "revision": "a" * 40, "source_sha256": "b" * 64,
            "pooling": POOLING, "dimensions": DIMENSIONS, "max_tokens": MAX_TOKENS,
            "shards": SHARDS, "frozen": True,
        }
        self.n = 51
        for shard in range(SHARDS):
            ids = np.array([str(i) for i in range(shard, self.n, SHARDS)])
            values = np.zeros((len(ids), DIMENSIONS), dtype=np.float32)
            values[:, 0] = 1.
            np.savez(self.root / f"{shard:02d}.npz", ids=ids, embeddings=values)
            (self.root / f"{shard:02d}.json").write_text(json.dumps({
                **self.manifest, "shard": shard, "rows": len(ids), "source_rows_scanned": self.n,
            }))

    def tearDown(self):
        self.temporary.cleanup()

    def test_merge_aligns_complete_source(self):
        ids, values, receipts = merge_shards(self.root, self.manifest)
        self.assertEqual(set(ids), {str(i) for i in range(self.n)})
        self.assertEqual(values.shape, (self.n, DIMENSIONS))
        self.assertEqual(len(receipts), 25)

    def test_revision_mismatch_is_rejected(self):
        path = self.root / "04.json"
        receipt = json.loads(path.read_text())
        receipt["revision"] = "c" * 40
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "revision"):
            merge_shards(self.root, self.manifest)

    def test_duplicate_row_identity_is_rejected(self):
        path = self.root / "01.npz"
        with np.load(path) as part:
            ids, values = part["ids"].copy(), part["embeddings"].copy()
        ids[0] = "0"
        np.savez(path, ids=ids, embeddings=values)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            merge_shards(self.root, self.manifest)

    def test_missing_shard_is_rejected(self):
        (self.root / "24.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            merge_shards(self.root, self.manifest)

    def test_corrupt_vector_is_rejected(self):
        path = self.root / "02.npz"
        with np.load(path) as part:
            ids, values = part["ids"].copy(), part["embeddings"].copy()
        values[0, 1] = np.nan
        np.savez(path, ids=ids, embeddings=values)
        with self.assertRaisesRegex(ValueError, "Invalid ELECTRA"):
            merge_shards(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
