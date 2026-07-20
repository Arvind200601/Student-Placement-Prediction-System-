import unittest

from app import app


class PlacementAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_renders(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_manual_page_renders(self):
        response = self.client.get('/manual')
        self.assertEqual(response.status_code, 200)

    def test_resume_page_renders(self):
        response = self.client.get('/resume')
        self.assertEqual(response.status_code, 200)

    def test_about_page_renders(self):
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
