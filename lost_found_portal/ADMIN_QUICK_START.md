# Quick Start Guide - Admin Panel

## Accessing the Admin Panel

### 1. Login as Admin or HOD
Use one of these default accounts:
- **Admin**: 
  - Registration No: `ADMIN001`
  - Password: `adminpassword`
  
- **HOD**:
  - Registration No: `HOD001`
  - Password: `hodpassword`

### 2. Navigate to Admin Panel
After logging in, you'll see "Admin Panel" in the navigation menu (between "Report Found" and "History").

Click on it or go to: `http://localhost:5000/admin`

## Admin Panel Features

### Dashboard Overview
The dashboard shows 4 quick stats at the top:
- Total Users
- Active Items  
- Total Transactions
- Archived Items

### Tab 1: Users Management
**Features:**
- View all users in the system
- Search users by name or registration number
- Filter by role (Student/Admin/HOD)
- View detailed user information
- Change user roles (Admin only)

**Actions:**
- Click "View" to see user details
- Click "Role" to change a user's role

### Tab 2: Transactions
**Features:**
- View all system transactions (last 100)
- Filter by action type:
  - Reported Lost
  - Reported Found
  - Deleted
  - Message Posted
  - Points Awarded
- Filter by date
- Click item ID to view item details

### Tab 3: Items
**Features:**
- View all items (active and archived)
- Filter by status (Lost/Found)
- Filter by state (Active/Archived)
- Quick access to item details

### Tab 4: Analytics
**Features:**
- User statistics (students, admins, HODs, average points)
- Item statistics (lost/found items, match rate, items this week)
- Top contributors leaderboard

## Enhanced History Page

### New Features
1. **Item ID Column**: Shows unique item identifier
2. **Image Preview**: Click thumbnails to view full size
3. **Blockchain Verification**: Click "Verify" to check blockchain integrity
4. **Better Filtering**: Filter by active/archived status

### Blockchain Verification
- Each item with a blockchain hash shows first 8 characters
- Click "Verify" button to check if item data has been tampered with
- ✅ Green checkmark = Verified authentic
- ❌ Red X = Data has been modified

## User Detail Page

Access by clicking "View" on any user in the Users tab.

**Information Shown:**
- User profile (registration number, name, email, role, points, level)
- Statistics (total reports, active items, total actions)
- All items reported by the user
- Activity logs (all actions performed)
- Authentication logs (login/logout history with IP addresses)

## Changing User Roles

**Requirements:**
- Must be logged in as Admin (not HOD)
- Only admins can change roles

**Steps:**
1. Go to Users tab
2. Click "Role" button next to the user
3. Enter new role: `student`, `admin`, or `hod`
4. Confirm the change
5. Page will reload with updated role

**Note:** All role changes are logged in the system for audit purposes.

## Security Notes

1. **Access Control**: Only Admin and HOD users can access the admin panel
2. **Role Changes**: Only Admin users can change roles (HOD cannot)
3. **Audit Trail**: All actions are logged in the transaction logs
4. **Authentication Tracking**: All logins/logouts are tracked with IP addresses

## Tips

1. **Search**: Use the search boxes to quickly find users or filter data
2. **Filters**: Combine multiple filters for precise results
3. **Mobile**: The admin panel is fully responsive and works on mobile devices
4. **Blockchain**: Regularly verify blockchain integrity for important items
5. **Analytics**: Check analytics tab for system insights and trends

## Troubleshooting

### Can't Access Admin Panel
- Make sure you're logged in as Admin or HOD
- Check your user role in the profile page
- Contact an admin to change your role if needed

### Role Change Not Working
- Only admins can change roles (not HOD)
- Make sure you're entering valid role: `student`, `admin`, or `hod`
- Check browser console for errors

### Blockchain Verification Fails
- This means the item data has been modified after blockchain recording
- Check the item detail page for more information
- Contact system administrator if this is unexpected

## Default Admin Credentials

**IMPORTANT**: Change these default passwords in production!

- Admin: ADMIN001 / adminpassword
- HOD: HOD001 / hodpassword

To change passwords, use the profile page or create new admin users.
